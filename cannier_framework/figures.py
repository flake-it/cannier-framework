import os
import numpy as np

from cannier_framework.globals import DATA, TABLES_DIR, PLOTS_DIR
from cannier_framework.technique import init_techniques, TECHNIQUES

from cannier_framework.model import (
    MODELS, BEST_MODELS, init_models_features, init_models_config
)

from cannier_framework.utils import (
    Table, get_sci_notation, load_subjects, PointPlot, get_project_names
)


class SubjectsTable(Table):
    def format_cell(self, x, y, cell):
        if y == 0: return cell
        elif cell == 0: return "-"
        elif y <= 5: return "%.0f" % cell
        else: return f"${get_sci_notation(cell)}$"


def make_subjects_table(subjects):
    n_proj = len(subjects)
    data = np.zeros((n_proj + 1, 6), dtype=float)
    data[:n_proj, 0] = DATA.item_counts

    data[:n_proj, 1:] = np.c_[
        DATA.project_masks @ DATA.items[:, (3, 4, 6)], DATA.total_pairs, 
        DATA.project_times
    ]

    data[n_proj] = data[:n_proj].sum(axis=0)
    SubjectsTable.from_data(subjects, data).make("subjects.tex")


class ConfigTable(Table):
    def format_cell(self, x, y, cell):
        if y == 3: return "%.3f" % cell
        else: return cell


def make_config_table(problem_id):
    perf = []

    for model_id, model in MODELS.items():
        (
            _problem_id, n_features, classifier_id, n_estimators, smote_id
        ) = model_id.split("_", 4)

        if problem_id != _problem_id or n_features != "18": continue

        perf.append([
            classifier_id, n_estimators, smote_id.replace("SMOTE+", "+"), 
            MODELS[model_id].get_scores_simple(None, .5, .5, 1)[0]
        ])

    perf.sort(key=lambda x: -x[3])
    ConfigTable(perf[:12]).make(f"{problem_id}_config.tex", False)


def make_features_plot():
    plots = []

    for (problem_id, model_id), mark in zip(
        BEST_MODELS.items(), ("+", "*", "x", "o")
    ):
        classifier_id, n_estimators, smote_id = model_id.split("_", 4)[2:]
        data = []

        for _model_id, model in MODELS.items():
            (
                _problem_id, n_features, _classifier_id, _n_estimators, 
                _smote_id
            ) = _model_id.split("_", 4)

            if (
                problem_id != _problem_id or classifier_id != _classifier_id or 
                n_estimators != _n_estimators or smote_id != _smote_id
            ):
                continue

            perf = MODELS[_model_id].get_scores_simple(None, .5, .5, 1)[0]
            data.append([int(n_features), perf])
            
        plots.append([f"only marks, mark={mark}", data, problem_id])

    PointPlot(plots).make("features.tex")


def make_figures():
    subjects = load_subjects()
    DATA.get_items_features(len(subjects))
    DATA.get_features_mean()
    DATA.get_rest(len(subjects))
    init_models_config()
    init_models_features()
    init_techniques()
    for model in MODELS.values(): model.load_preds()
    os.makedirs(TABLES_DIR, exist_ok=True)
    make_subjects_table(subjects)
    for problem_id in BEST_MODELS: make_config_table(problem_id)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    make_features_plot()
    project_names = get_project_names(subjects)

    for model_id in BEST_MODELS.values():
        model = MODELS[model_id]
        model.make_shap_plot()
        model.make_box_plot()
        model.make_point_plot()
        model.make_table(project_names)

    for technique in TECHNIQUES.values():
        points = technique.load_points()
        pin_points = technique.get_pin_points(points)
        technique.make_plot(points, pin_points)
        technique.make_table(project_names, *pin_points[0, 2:])
