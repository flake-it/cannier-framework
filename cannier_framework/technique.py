import os
import gc
import random
import numpy as np

from multiprocessing import Pool

from cannier_framework.globals import ARGS, POINTS_DIR, DATA
from cannier_framework.utils import get_sci_notation, PointPlot, load_subjects

from cannier_framework.model import (
    ModelTable, MODELS, BEST_MODELS, init_models_best
)


def get_points_chunk(args):
    gc.enable()
    technique_id, chunk = args
    return TECHNIQUES[technique_id].get_points_chunk(chunk)


def get_pareto_front(scores):
    front = np.arange(scores.shape[0])
    i = 0

    while i < scores.shape[0]:
        nondominated = np.any(scores < scores[i], axis=1)
        nondominated[i] = True
        scores = scores[nondominated]
        front = front[nondominated]
        i = np.sum(nondominated[:i]) + 1

    return front


class Technique:
    def __init__(self):
        raise NotImplementedError

    def iter_params(self):
        raise NotImplementedError

    def eval_params(self, thresh1, thresh2, n_samples):
        raise NotImplementedError

    def get_points_chunk(self, chunk):
        points_chunk = {}

        for thresh1, thresh2, n_samples in chunk:
            perf, cost = self.eval_params(thresh1, thresh2, n_samples)

            if np.isfinite(perf):
                points_chunk[(perf, cost)] = thresh1, thresh2, n_samples

        return points_chunk

    def get_points(self):
        params = list(self.iter_params())
        random.shuffle(params)

        args = [
            (self.technique_id, chunk) 
            for chunk in np.array_split(params, ARGS.processes)
        ]

        points = {}
        gc.disable()

        with Pool(processes=ARGS.processes) as pool:
            for points_chunk in pool.imap_unordered(get_points_chunk, args):
                points.update(points_chunk)
        
        return np.array(
            [[*scores, *params] for scores, params in points.items()], 
            dtype=float
        )

    def get_points_file(self):
        return os.path.join(POINTS_DIR, f"{self.technique_id}.npy")

    def make_points(self):
        points = self.get_points()
        points[:, 0] = 1 - points[:, 0]
        points = points[get_pareto_front(points[:, :2])]
        points[:, 0] = 1 - points[:, 0]
        np.save(self.get_points_file(), points[points[:, 0].argsort()])

    def load_points(self):
        return np.load(self.get_points_file())

    def get_pin_points(self, points):
        raise NotImplementedError

    def get_scores_label(self, perf, cost):
        perf = "%.2f" % perf
        cost = get_sci_notation(cost)
        return f"[{perf}, {cost}]"

    def get_params_label(self, thresh1, thresh2, n_samples):
        raise NotImplementedError

    def make_plot(self, points, pin_points):
        pins = []

        for pp in pin_points:
            pins.append([*pp[:2], 135, self.get_scores_label(*pp[:2])])
            pins.append([*pp[:2], 315, self.get_params_label(*pp[2:])])

        plot = PointPlot([("no marks", points[:, :2], "")], pins)
        plot.make(f"{self.technique_id}.tex")

    def make_table(self, project_names, thresh1, thresh2, n_samples):
        raise NotImplementedError


def get_knee_point(points):
    scores = points[:, :2].copy()
    scores -= scores.min(axis=0)
    scores /= scores.max(axis=0)
    scores -= np.array([1, 0])
    return points[np.linalg.norm(scores, axis=1).argmin()]


class SingleModelTechnique(Technique):
    def iter_params(self):
        for thresh_l in range(101):
            for thresh_u in range(thresh_l, 102):
                if thresh_l == 0 and thresh_u == 101:
                    yield 0, 1.01, 0
                else:
                    for n_samples in range(1, 16):
                        yield .01 * thresh_l, .01 * thresh_u, n_samples

    def eval_params(self, thresh_l, thresh_u, n_samples):
        return self.model.get_scores_simple(
            self.item_costs, thresh_l, thresh_u, int(n_samples)
        )

    def get_pin_points(self, points):
        all_model, no_model = [
            masked[np.argmax(masked[:, 0])] for masked in [
                points[(points[:, 2] == points[:, 3])],
                points[(points[:, 2] == 0) * (points[:, 3] == 1.01)]
            ]
        ]

        return np.array([get_knee_point(points), all_model, no_model])

    def get_params_label(self, thresh_l, thresh_u, n_samples):
        thresh_l = "\\omega_l = %.2f" % thresh_l
        thresh_u = "\\omega_u = %.2f" % thresh_u
        n_samples = "n_F = %.0f" % n_samples
        return f"({thresh_l}, {thresh_u}, {n_samples})"

    def make_table(self, project_names, thresh_l, thresh_u, n_samples):
        scores = self.model.get_scores_full(
            self.item_costs, thresh_l, thresh_u, int(n_samples)
        )

        full_costs = self.model.project_masks @ self.item_costs
        scores = np.c_[scores, np.r_[full_costs, full_costs.sum()]]
        table = ModelTable.from_data(project_names, scores)
        table.make(f"{self.technique_id}.tex")


def get_exp_reruns(p_fail, n_reruns):
    aa = p_fail ** (n_reruns + 1)
    ab = (p_fail - 1) * (1 - p_fail) ** n_reruns
    ac = p_fail ** 2
    a = aa - ab - ac + p_fail - 1
    b = (p_fail - 1) * p_fail
    return a / b


class Rerun(SingleModelTechnique):
    def __init__(self):
        self.technique_id = "Rerun"
        self.model = MODELS[BEST_MODELS["NOD-vs-Rest"]]
        labels = self.model.labels
        p_fail = DATA.items[labels, 1] / ARGS.n_reruns
        exp_reruns = get_exp_reruns(p_fail, ARGS.n_reruns)
        self.item_costs = DATA.item_times.copy()
        self.item_costs[labels] *= exp_reruns
        self.item_costs[~labels] *= ARGS.n_reruns


class iDFlakies(SingleModelTechnique):
    def __init__(self):
        self.technique_id = "iDFlakies"
        self.model = MODELS[BEST_MODELS["NOD-vs-Victim"]]
        items = self.model.items
        n_reruns = 1 + .2 * (items[:, 2] - 1)
        self.item_costs = DATA.project_times[items[:, 0]] * n_reruns


class PairwiseTable(ModelTable):
    def format_cell(self, x, y, cell):
        if y == 0: return cell
        elif cell == 0: return "-"
        elif np.isnan(cell) or not np.isfinite(cell): return "$\\bot$"
        elif y <= 2: return "%.0f" % cell
        elif y == 3: return "%.2f" % cell
        else: return f"${get_sci_notation(cell)}$"


class Pairwise(Technique):
    def __init__(self):
        self.technique_id = "Pairwise"
        self.model_v = MODELS[BEST_MODELS["Victim-vs-Rest"]]
        self.model_p = MODELS[BEST_MODELS["Polluter-vs-Rest"]]

    def iter_params(self):
        for thresh_v in range(101):
            for thresh_p in range(101):
                if thresh_v == thresh_p == 0:
                    yield 0, 0, 0
                else:
                    for n_samples in range(1, 16):
                        yield .01 * thresh_v, .01 * thresh_p, n_samples

    def get_scores_ij(self, thresh_v, thresh_p, n_samples, i, j):
        scores_ij = np.empty((DATA.project_masks.shape[0], 2), dtype=float)
        preds_v = self.model_v.preds[n_samples][i]
        preds_p = self.model_p.preds[n_samples][j]

        for proj_id, scores_proj in enumerate(scores_ij):
            dependencies = DATA.dependencies[proj_id]
            mask_proj = DATA.project_masks[proj_id]
            mask_v = preds_v[mask_proj] >= thresh_v
            mask_p = preds_p[mask_proj] >= thresh_p
            scores_proj[0] = dependencies[mask_v][:, mask_p].sum()
            item_times = DATA.item_times[mask_proj]
            scores_proj[1] = mask_p.sum() * item_times[mask_v].sum()
            scores_proj[1] += mask_v.sum() * item_times[mask_p].sum()

        return scores_ij

    def get_scores(self, thresh_v, thresh_p, n_samples):
        n_proj = DATA.project_masks.shape[0]
        scores = np.zeros((n_proj + 1, 4), dtype=float)
        scores[:n_proj, 1] = DATA.total_pairs

        if n_samples > 0:
            rng = np.random.default_rng(seed=0)

            for i, j in rng.choice(ARGS.n_repeats, (ARGS.n_repeats, 2)):
                scores[:n_proj, (0, 3)] += self.get_scores_ij(
                    thresh_v, thresh_p, n_samples, i, j
                )

            scores[:n_proj, (0, 3)] /= ARGS.n_repeats
            scores[:n_proj, 3] += n_samples * DATA.project_times
        else:
            scores[:n_proj, 0] = DATA.total_pairs
            scores[:n_proj, 3] = 2 * DATA.item_counts * DATA.project_times

        scores[n_proj] = scores[:n_proj].sum(axis=0)
        scores[:n_proj, 2] = scores[:n_proj, 0] / DATA.total_pairs
        scores[n_proj, 2] = scores[n_proj, 0] / DATA.total_pairs.sum()
        return scores

    def eval_params(self, thresh_v, thresh_p, n_samples):
        scores = self.get_scores(thresh_v, thresh_p, int(n_samples))
        return scores[-1, 2:]

    def get_pin_points(self, points):
        return np.array([
            get_knee_point(points), points[np.argmax(points[:, 0])]
        ])

    def get_params_label(self, thresh_v, thresh_p, n_samples):
        thresh_v = "\\omega_V = %.2f" % thresh_v
        thresh_p = "\\omega_P = %.2f" % thresh_p
        n_samples = "n_F = %.0f" % n_samples
        return f"({thresh_v}, {thresh_p}, {n_samples})"

    def make_table(self, project_names, thresh_v, thresh_p, n_samples):
        scores = self.get_scores(thresh_v, thresh_p, int(n_samples))
        full_costs = 2 * DATA.item_counts * DATA.project_times
        scores = np.c_[scores, np.r_[full_costs, full_costs.sum()]]
        table = PairwiseTable.from_data(project_names, scores)
        table.make(f"{self.technique_id}.tex")


TECHNIQUES = {}


def init_techniques():
    for technique in (Rerun(), iDFlakies(), Pairwise()):
        TECHNIQUES[technique.technique_id] = technique


def make_points():
    n_proj = len(load_subjects())
    DATA.get_items_features(n_proj)
    DATA.get_features_mean()
    DATA.get_rest(n_proj)
    init_models_best()
    init_techniques()
    for model in MODELS.values(): model.load_preds()
    os.makedirs(POINTS_DIR, exist_ok=True)
    for technique in TECHNIQUES.values(): technique.make_points()
