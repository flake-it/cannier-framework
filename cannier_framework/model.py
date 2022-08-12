import os
import shap
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as pl

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

from cannier_framework.globals import (
    DATA, SHAP_DIR, PLOTS_DIR, ARGS, PREDS_DIR
)

from cannier_framework.utils import (
    Table, get_sci_notation, PointPlot, BoxPlot, load_subjects, manage_pool
)


FEATURE_NAMES = (
    "Read Count", "Write Count", "Run Time", "Wait Time", "Context Switches",
    "Covered Lines", "Source Covered Lines", "Covered Changes", "Max. Threads",
    "Max. Children", "Max. Memory", "AST Depth", "Assertions", 
    "External Modules", "Halstead Volume", "Cyclomatic Complexity", 
    "Test Lines of Code", "Maintainability"
)


def get_mcc(tn, fn, fp, tp):
    a = tp * tn - fp * fn
    b = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return np.abs(a / np.sqrt(b))


class ModelTable(Table):
    def keep_row(self, row):
        return all(np.isfinite(cell) for cell in row)

    def format_cell(self, x, y, cell):
        if cell == 0:
            return "-"
        elif y <= 3:
            return "%.0f" % cell
        elif y == 4:
            return "%.2f" % cell
        elif y == 5:
            return f"${get_sci_notation(cell)}$"


class Model:
    def __init__(self, model_id, domain_idx, labels_idx):
        self.model_id = model_id
        self.domain_idx = domain_idx
        self.labels_idx = labels_idx

    def setup(self):
        if self.domain_idx is None:
            self.domain_mask = None
            self.items = DATA.items
            self.features = DATA.features
            self.features_mean = DATA.features_mean
            self.project_masks = DATA.project_masks
        else:
            self.domain_mask = DATA.items[:, self.domain_idx].astype(bool)
            self.items = DATA.items[self.domain_mask]
            self.features = DATA.features[self.domain_mask]
            self.features_mean = DATA.features_mean[self.domain_mask]
            self.project_masks = DATA.project_masks[:, self.domain_mask]

        self.labels = self.items[:, self.labels_idx].astype(bool)

    def get_shap_file(self):
        return os.path.join(SHAP_DIR, f"{self.model_id}.npy")

    def make_shap(self):
        classifier = ExtraTreesClassifier()
        classifier.fit(*SMOTE().fit_resample(self.features_mean, self.labels))
        explainer = shap.TreeExplainer(classifier)
        positive = list(classifier.classes_).index(True)
        shap_values = explainer.shap_values(self.features_mean)[positive]

        with open(self.get_shap_file(), "wb") as f:
            np.save(f, shap_values)

        return self.model_id, True

    def make_shap_plot(self):
        with open(self.get_shap_file(), "rb") as f:
            shap_values = np.load(f)
            
        shap_values_ma = np.abs(shap_values).mean(axis=0)
        order = np.argsort(-shap_values_ma)
        
        feature_names = [
            "%s (%0.3f)" % (FEATURE_NAMES[i], shap_values_ma[i]) 
            for i in order
        ]

        plot_file = os.path.join(PLOTS_DIR, f"{self.model_id}_shap.pdf")
        pl.rc("font", family="serif")

        shap.summary_plot(
            shap_values[:, order], features=self.features_mean[:, order], 
            feature_names=feature_names, show=False, sort=False, 
            color_bar=False, plot_size=(7, 8)
        )

        pl.xlim([-.3, .3])
        pl.xlabel("SHAP values", fontsize=16)
        pl.gca().tick_params("x", labelsize=16)
        pl.gca().tick_params("y", labelsize=16)
        pl.tight_layout()
        pl.savefig(plot_file, format="pdf")
        pl.clf()

    def set_preds_i(self, n_samples, preds_i):
        fold = StratifiedKFold(n_splits=10, shuffle=True)
        choice = np.random.choice(ARGS.n_repeats, n_samples, replace=False)
        features = self.features[:, choice].mean(axis=1)
        balancing = SMOTE()
        classifier = ExtraTreesClassifier()

        for train, test in fold.split(features, self.labels):
            features_train, labels_train = balancing.fit_resample(
                features[train], self.labels[train]
            )

            classifier.fit(features_train, labels_train)
            proba = classifier.predict_proba(features[test])
            positive = list(classifier.classes_).index(True)
            preds_i[test] = proba[:, positive]

    def get_preds_n_file(self, n_samples):
        return os.path.join(PREDS_DIR, f"{self.model_id}_{n_samples}.npy")

    def make_preds_n(self, n_samples):
        n_items = self.items.shape[0]
        preds_n = np.empty((ARGS.n_repeats, n_items), dtype=float)

        for preds_i in preds_n:
            self.set_preds_i(n_samples, preds_i)

        np.save(self.get_preds_n_file(n_samples), preds_n)
        return f"{self.model_id}_{n_samples}", True

    def load_preds(self):
        n_items = self.items.shape[0]
        self.preds = np.empty((15, ARGS.n_repeats, n_items), dtype=float)

        for n_samples in range(1, 16):
            preds_n = np.load(self.get_preds_n_file(n_samples))
            self.preds[n_samples - 1] = preds_n

    def get_confusion(self, thresh_l, thresh_u, n_samples, i):
        categories = np.zeros((self.labels.shape[0], 4))
        preds_i = self.preds[n_samples - 1, i]
        neg_mask = preds_i < thresh_l
        categories[neg_mask, self.labels[neg_mask] + 0] = 1
        pos_mask = preds_i >= thresh_u
        categories[pos_mask, self.labels[pos_mask] + 2] = 1
        amb_mask = ~(neg_mask + pos_mask)
        neg_amb_mask = amb_mask * ~self.labels
        categories[neg_amb_mask, 0] = 1
        pos_amb_mask = amb_mask * self.labels
        categories[pos_amb_mask, 3] = 1
        return categories, amb_mask

    def make_box_plot(self):
        counts = np.zeros(4)
        plot = np.zeros([4, 5])

        for i in range(ARGS.n_repeats):
            categories = self.get_confusion(.5, .5, 1, i)[0].astype(bool)
            counts += categories.sum(axis=0)

            for j in range(4):
                masked = self.preds[0, i][categories[:, j]]

                for k, q in enumerate([5, 25, 50, 75, 95]):
                    plot[j, k] += np.percentile(masked, q)

        info = self.model_id, *(counts / ARGS.n_repeats)
        print("%s: TN = %.0f, FN = %.0f, FP = %.0f, TP = %.0f" % info)
        BoxPlot(plot / ARGS.n_repeats).make(f"{self.model_id}_box.tex")
        
    def get_scores_simple_i(
        self, item_costs, thresh_l, thresh_u, n_samples, i
    ):
        categories, amb_mask = self.get_confusion(
            thresh_l, thresh_u, n_samples, i
        )

        perf = get_mcc(*categories.sum(axis=0))
        cost = 0 if item_costs is None else item_costs[amb_mask].sum()
        return np.array([perf, cost])

    def get_scores_simple(self, item_costs, thresh_l, thresh_u, n_samples):
        if n_samples > 0:
            scores = np.zeros(2, dtype=float)

            for i in range(ARGS.n_repeats):
                scores += self.get_scores_simple_i(
                    item_costs, thresh_l, thresh_u, n_samples, i
                )

            scores /= ARGS.n_repeats
            scores[1] += n_samples * DATA.project_times.sum()
        else:
            cost = 0 if item_costs is None else item_costs.sum()
            scores = np.array([1, cost], dtype=float)
            
        return scores

    def make_point_plot(self):
        data = np.empty((15, 2), dtype=float)

        for n_samples in range(1, 16):
            perf, _ = self.get_scores_simple(None, .5, .5, n_samples)
            data[n_samples - 1] = n_samples, perf

        slope, intercept = np.polyfit(data[:, 0], data[:, 1], 1)
        correlation = ss.spearmanr(data).correlation
        info = self.model_id, slope, intercept, correlation
        print("%s: slope = %.4f, intercept = %.2f, correlation = %.2f" % info)
        line = np.array([[x, slope * x + intercept] for x in range(1, 16)])
        plot = PointPlot([("only marks, mark=x", data), ("thick, red", line)])
        plot.make(f"{self.model_id}_point.tex")

    def get_scores_full_i(self, item_costs, thresh_l, thresh_u, n_samples, i):
        n_proj = self.project_masks.shape[0]
        scores_i = np.zeros((n_proj + 1, 6), dtype=float)

        categories, amb_mask = self.get_confusion(
            thresh_l, thresh_u, n_samples, i
        )

        scores_i[:n_proj, :4] = self.project_masks @ categories

        if item_costs is not None:
            scores_i[:n_proj, 5] = self.project_masks @ (amb_mask * item_costs)

        scores_i[n_proj] = scores_i[:n_proj].sum(axis=0)
        scores_i[:, 4] = get_mcc(*scores_i[:, :4].T)
        return scores_i

    def get_scores_full(self, item_costs, thresh_l, thresh_u, n_samples):
        n_proj = self.project_masks.shape[0]

        if n_samples > 0:
            scores = np.zeros((n_proj + 1, 6), dtype=float)

            for i in range(ARGS.n_repeats):
                scores += self.get_scores_full_i(
                    item_costs, thresh_l, thresh_u, n_samples, i
                )

            scores /= ARGS.n_repeats
            scores[:n_proj, 5] += n_samples * DATA.project_times
            scores[n_proj, 5] += n_samples * DATA.project_times.sum()
        else:
            scores = np.zeros((n_proj + 1, 6), dtype=float)
            labels = self.labels.astype(int)
            scores[:n_proj, 0] = self.project_masks @ labels
            scores[:n_proj, 3] = self.project_masks @ (1 - labels)

            if item_costs is not None:
                scores[:n_proj, 5] = self.project_masks @ item_costs

            scores[n_proj] = scores[:n_proj].sum(axis=0)
            scores[:, 4] = 1

        return scores

    def make_table(self, project_names):
        scores = self.get_scores_full(None, .5, .5, 1)
        mean, std = np.nanmean(scores[:-1, 4]), np.nanstd(scores[:-1, 4])
        print("%s: mean = %.2f, std = %.2f" % (self.model_id, mean, std))
        table = ModelTable.from_data(project_names, scores[:, :5])
        table.make(f"{self.model_id}.tex")


MODELS = {
    model.model_id: model for model in (
        Model("NOD-vs-Rest", None, 3), Model("NOD-vs-Victim", 5, 3), 
        Model("Victim-vs-Rest", None, 4), Model("Polluter-vs-Rest", None, 6)
    )
}


def make_shap_model(model_id):
    return MODELS[model_id].make_shap()

    
def make_shap():
    DATA.setup(len(load_subjects()))

    for model in MODELS.values():
        model.setup()

    os.makedirs(SHAP_DIR, exist_ok=True)
    manage_pool(make_shap_model, MODELS)


def make_preds_model_n(args):
    model_id, n_samples = args
    return MODELS[model_id].make_preds_n(n_samples)


def make_preds():
    DATA.setup(len(load_subjects()))
    args = []

    for model_id, model in MODELS.items():
        model.setup()

        for n_samples in range(1, 16):
            args.append([model_id, n_samples])

    os.makedirs(PREDS_DIR, exist_ok=True)
    manage_pool(make_preds_model_n, args)
