import gc
import os
import shap
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as pl

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

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
    def format_cell(self, x, y, cell):
        if y == 0: return cell
        elif cell == 0: return "-"
        elif np.isnan(cell) or not np.isfinite(cell): return "$\\bot$"
        elif y <= 4: return "%.0f" % cell
        elif y == 5: return "%.2f" % cell
        else: return f"${get_sci_notation(cell)}$"


class Model:
    def __init__(
        self, model_id, domain_idx, labels_idx, features_mask, classifier_cls, 
        n_estimators, smote_cls
    ):
        self.model_id = model_id

        (
            self.items, self.features, self.features_mean, self.project_masks, 
            self.labels
        ) = DATA.get_model_data(domain_idx, labels_idx, features_mask)

        self.classifier_cls = classifier_cls
        self.n_estimators = n_estimators
        self.smote_cls = smote_cls

    def get_shap_file(self):
        return os.path.join(SHAP_DIR, f"{self.model_id}.npy")

    def make_shap(self):
        classifier = self.classifier_cls(n_estimators=self.n_estimators)
        smote = self.smote_cls()
        classifier.fit(*smote.fit_resample(self.features_mean, self.labels))
        explainer = shap.TreeExplainer(classifier)
        positive = list(classifier.classes_).index(True)
        shap_values = explainer.shap_values(self.features_mean)[positive]
        with open(self.get_shap_file(), "wb") as f: np.save(f, shap_values)
        return self.model_id, True

    def make_shap_plot(self):
        with open(self.get_shap_file(), "rb") as f: shap_values = np.load(f)
        shap_values_ma = np.abs(shap_values).mean(axis=0)
        order = np.argsort(-shap_values_ma)

        feature_names = [
            "%s (%0.3f)" % (FEATURE_NAMES[i], shap_values_ma[i]) for i in order
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
        classifier = self.classifier_cls(n_estimators=self.n_estimators)
        smote = self.smote_cls()

        for train, test in fold.split(features, self.labels):
            features_train, labels_train = smote.fit_resample(
                features[train], self.labels[train]
            )

            classifier.fit(features_train, labels_train)
            proba = classifier.predict_proba(features[test])
            positive = list(classifier.classes_).index(True)
            preds_i[test] = proba[:, positive]

    def get_preds_n_file(self, n_samples):
        return os.path.join(PREDS_DIR, f"{self.model_id}_{n_samples}.npy")

    def make_preds_n(self, n_samples):
        preds_n = np.empty((ARGS.n_repeats, self.items.shape[0]), dtype=float)
        for preds_i in preds_n: self.set_preds_i(n_samples, preds_i)
        np.save(self.get_preds_n_file(n_samples), preds_n)
        return f"{self.model_id}_{n_samples}", True

    def load_preds(self):
        self.preds = {}

        for n_samples in range(1, 16):
            try: preds_n = np.load(self.get_preds_n_file(n_samples))
            except FileNotFoundError: continue
            self.preds[n_samples] = preds_n

    def get_confusion(self, thresh_l, thresh_u, n_samples, i):
        categories = np.zeros((self.labels.shape[0], 4))
        preds_i = self.preds[n_samples][i]
        neg_mask = preds_i < thresh_l
        categories[neg_mask, self.labels[neg_mask] + 0] = 1
        pos_mask = preds_i >= thresh_u
        categories[pos_mask, self.labels[pos_mask] + 2] = 1
        amb_mask = ~(neg_mask + pos_mask)
        categories[amb_mask * ~self.labels, 0] = 1
        categories[amb_mask * self.labels, 3] = 1
        return categories, amb_mask

    def make_box_plot(self):
        plot = np.zeros([4, 5])
        counts = np.zeros(4)

        for i in range(ARGS.n_repeats):
            categories = self.get_confusion(.5, .5, 1, i)[0].astype(bool)
            counts += categories.sum(axis=0)

            for j in range(4):
                for k, q in enumerate([5, 25, 50, 75, 95]):
                    plot[j, k] += np.percentile(
                        self.preds[1][i][categories[:, j]], q
                    )

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
        scores = np.zeros(2, dtype=float)

        if n_samples > 0:
            for i in range(ARGS.n_repeats):
                scores += self.get_scores_simple_i(
                    item_costs, thresh_l, thresh_u, n_samples, i
                )

            scores /= ARGS.n_repeats
            scores[1] += n_samples * DATA.project_times.sum()
        else:
            scores[0] = 1
            scores[1] = 0 if item_costs is None else item_costs.sum()
            
        return scores

    def make_point_plot(self):
        data = np.empty((15, 2), dtype=float)

        for n_samples in range(1, 16):
            perf = self.get_scores_simple(None, .5, .5, n_samples)[0]
            data[n_samples - 1] = n_samples, perf

        slope, intercept = np.polyfit(data[:, 0], data[:, 1], 1)
        info = self.model_id, slope, intercept, ss.spearmanr(data).correlation
        print("%s: slope = %.4f, intercept = %.2f, correlation = %.2f" % info)
        line = np.array([[x, slope * x + intercept] for x in range(1, 16)])
        
        plot = PointPlot([
            ("only marks, mark=x", data, ""), ("thick, red", line, "")
        ])

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
        scores = np.zeros((n_proj + 1, 6), dtype=float)

        if n_samples > 0:
            for i in range(ARGS.n_repeats):
                scores += self.get_scores_full_i(
                    item_costs, thresh_l, thresh_u, n_samples, i
                )

            scores /= ARGS.n_repeats
            scores[:n_proj, 5] += n_samples * DATA.project_times
            scores[n_proj, 5] += n_samples * DATA.project_times.sum()
        else:
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


BEST_MODELS = {
    "NOD-vs-Rest": "NOD-vs-Rest_18_ExtraTrees_100_SMOTE",
    "NOD-vs-Victim": "NOD-vs-Victim_18_RandomForest_75_SMOTE",
    "Victim-vs-Rest": "Victim-vs-Rest_18_ExtraTrees_75_SMOTE",
    "Polluter-vs-Rest": "Polluter-vs-Rest_18_RandomForest_100_SMOTE"
}

MODELS = {}

PROBLEMS = {
    "NOD-vs-Rest": (None, 3), "NOD-vs-Victim": (5, 3), 
    "Victim-vs-Rest": (None, 4), "Polluter-vs-Rest": (None, 6)
}

CLASSIFIERS = {
    "RandomForest": RandomForestClassifier, "ExtraTrees": ExtraTreesClassifier
}

SMOTES = {"SMOTE": SMOTE, "SMOTE+ENN": SMOTEENN, "SMOTE+Tomek": SMOTETomek}

N_ESTIMATORS = 25, 50, 75, 100

FEATURES = {
    "NOD-vs-Rest": (
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        (0, 1, 2, 3, 4, 5, 8, 10, 11),
        (0, 2, 4, 8, 10, 11),
        (2, 8, 11)
    ),
    "NOD-vs-Victim": (
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15),
        (0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 15),
        (0, 1, 2, 4, 7, 8, 10, 13, 15),
        (0, 1, 2, 4, 8, 10),
        (2, 4, 8)
    ),
    "Victim-vs-Rest": (
        (0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17),
        (0, 1, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17),
        (1, 5, 6, 10, 11, 12, 13, 15, 17),
        (5, 10, 11, 12, 13, 17),
        (10, 11, 17)
    ),
    "Polluter-vs-Rest": (
        (0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17),
        (0, 1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14),
        (0, 1, 2, 5, 6, 8, 10, 12, 14),
        (1, 2, 5, 8, 10, 12),
        (2, 8, 10)
    )
}


def init_models_best():
    for model_id in BEST_MODELS.values():
        (
            problem_id, _, classifier_id, n_estimators, smote_id
        ) = model_id.split("_", 4)

        MODELS[model_id] = Model(
            model_id, *PROBLEMS[problem_id], None, CLASSIFIERS[classifier_id], 
            int(n_estimators), SMOTES[smote_id]
        )


def init_models_config():
    for problem_id, (domain_idx, labels_idx) in PROBLEMS.items():
        for classifier_id, classifier_cls in CLASSIFIERS.items():
            for n_estimators in N_ESTIMATORS:
                for smote_id, smote_cls in SMOTES.items():
                    model_id = "_".join([
                        problem_id, "18", classifier_id, str(n_estimators), 
                        smote_id
                    ])

                    MODELS[model_id] = Model(
                        model_id, domain_idx, labels_idx, None, classifier_cls, 
                        n_estimators, smote_cls
                    )


def init_models_features():
    for problem_id, model_id in BEST_MODELS.items():
        (
            problem_id, _, classifier_id, n_estimators, smote_id
        ) = model_id.split("_")

        for features_mask in FEATURES[problem_id]:
            model_id = "_".join([
                problem_id, str(len(features_mask)), classifier_id, 
                n_estimators, smote_id
            ])

            MODELS[model_id] = Model(
                model_id, *PROBLEMS[problem_id], features_mask, 
                CLASSIFIERS[classifier_id], int(n_estimators), SMOTES[smote_id]
            )


def make_shap_model(model_id):
    gc.enable()
    return MODELS[model_id].make_shap()

    
def make_shap():
    DATA.get_items_features(len(load_subjects()))
    DATA.get_features_mean()
    init_models_best()
    os.makedirs(SHAP_DIR, exist_ok=True)
    manage_pool(make_shap_model, MODELS)


def make_preds_model_n(args):
    gc.enable()
    model_id, n_samples = args
    return MODELS[model_id].make_preds_n(n_samples)


MENU = {
    "best": init_models_best,
    "config": init_models_config,
    "features": init_models_features
}


def make_preds(category):
    try:
        init_models = MENU[category]
    except KeyError:
        print(f"cannier-framework: '{ARGS.args[0]}' is not a valid category.")
        sys.exit(1)

    DATA.get_items_features(len(load_subjects()))
    init_models()
    start, stop = (2, 16) if category == "best" else (1, 2)
    args = []

    for model_id, model in MODELS.items():
        for n_samples in range(start, stop): args.append([model_id, n_samples])

    os.makedirs(PREDS_DIR, exist_ok=True)
    manage_pool(make_preds_model_n, args)
