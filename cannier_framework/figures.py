import os
import numpy as np

from cannier_framework.model import MODELS
from cannier_framework.technique import TECHNIQUES
from cannier_framework.globals import DATA, TABLES_DIR, PLOTS_DIR

from cannier_framework.utils import (
    Table, get_sci_notation, load_subjects, PointPlot, get_project_names
)


class SubjectsTable(Table):
    def format_cell(self, x, y, cell):
        if cell == 0:
            return "-"
        elif y <= 4:
            return "%.0f" % cell
        elif y == 5:
            return f"${get_sci_notation(cell)}$"


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


def make_figures():
    subjects = load_subjects()
    DATA.setup(len(subjects))
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    make_subjects_table(subjects)
    project_names = get_project_names(subjects)

    for model_id, model in MODELS.items():
        model.setup()
        model.load_preds()
        model.make_shap_plot()
        model.make_box_plot()
        model.make_point_plot()
        model.make_table(project_names)

    for technique_id, technique in TECHNIQUES.items():
        technique.setup()
        points = technique.load_points()
        pin_points = technique.get_pin_points(points)
        technique.make_plot(points, pin_points)
        technique.make_table(project_names, *pin_points[0, 2:])
