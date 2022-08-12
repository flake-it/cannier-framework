import os
import sys
import json
import time
import random

from multiprocessing import Pool

from cannier_framework.globals import ARGS, PLOTS_DIR, TABLES_DIR


def load_subjects():
    with open("subjects.json", "r") as f:
        return json.load(f)


def get_counters_max():
    return {
        "churn": 1, "features": ARGS.n_repeats, 
        "baseline": ARGS.n_reruns, "shuffle": ARGS.n_reruns
    }


def fetch_counters(cur):
    plugin_modes = "churn", "features", "baseline", "shuffle"

    cur.execute(
        "select count_churn, count_features, count_baseline, count_shuffle "
        "from counters "
        "where id = 1"
    )

    return dict(zip(plugin_modes, cur.fetchone()))


def manage_pool(fn, args):
    args = list(args)
    n_finish = 0
    all_success = True
    t_start = time.time()

    random.shuffle(args)
    sys.stdout.write(f"0/{len(args)} 0/?\r")

    with Pool(processes=min(ARGS.processes, len(args))) as pool:
        for message, success in pool.imap_unordered(fn, args):
            n_finish += 1
            all_success &= success
            n_remain = len(args) - n_finish
            t_elapse = time.time() - t_start
            t_remain = t_elapse / n_finish * n_remain
            t_elapse = round(t_elapse / 60)
            t_remain = round(t_remain / 60)

            sys.stdout.write(f"{message}\n\r")
            sys.stdout.write(f"{n_finish}/{n_remain} {t_elapse}/{t_remain}\r")

    return all_success


def load_training_data(n_proj):
    items = np.load(ITEMS_FILE)
    features = np.load(FEATURES_FILE)

    if n_proj != np.unique(items[:, 0]).shape[0]:
        print("cannier-framework: project count mismatch.")
        sys.exit(1)

    return items, features


class PointPlot:
    def __init__(self, plot, pins=()):
        self.plot = plot
        self.pins = pins

    def make(self, plot_file):
        with open(os.path.join(PLOTS_DIR, plot_file), "w") as f:
            for options, data in self.plot:
                if options:
                    f.write(f"\\addplot[{options}] ")
                else:
                    f.write(f"\\addplot ")

                coordinates = " ".join([f"({x},{y})" for x, y in data])
                f.write(f"coordinates {{{coordinates}}};\n")

            for x, y, angle, label in self.pins:
                f.write(
                    f"\\addplot[mark=*] coordinates {{({x}, {y})}} "
                    f"node [pin={angle}:{{${label}$}}]{{}};\n"
                )


class BoxPlot:
    def __init__(self, plot):
        self.plot = plot

    def make(self, plot_file):
        with open(os.path.join(PLOTS_DIR, plot_file), "w") as f:
            for data in self.plot:
                f.write("\\addplot+")

                data = zip(
                    [
                        "lower whisker", "lower quartile", "median", 
                        "upper quartile", "upper whisker"
                    ], 
                    data
                )

                data = ", ".join(["=".join([x, str(y)]) for x, y in data])
                f.write(f"[color=black, boxplot prepared={{{data}}}] ")
                f.write("coordinates {};\n")


class Table:
    def __init__(self, table):
        self.table = table

    @classmethod
    def from_data(cls, row_labels, data):
        return cls([
            [proj, *data_proj] 
            for proj, data_proj in zip([*row_labels, "{\\bf Dataset}"], data)
        ])

    def keep_row(self, row):
        return True

    def format_cell(self, x, y, cell):
        return str(cell)

    def make(self, table_file, total_row=True):
        table = [row for row in self.table if self.keep_row(row[1:])]

        with open(os.path.join(TABLES_DIR, table_file), "w") as f:
            for x, row in enumerate(table):
                if total_row and x == len(table) - 1:
                    f.write("\\midrule\n")
                elif x % 2 == 1:
                    f.write("\\rowcolor{gray!20}\n")

                row = [
                    cell if y == 0 else self.format_cell(x, y - 1, cell) 
                    for y, cell in enumerate(row)
                ]

                f.write(" & ".join(row) + " \\\\\n")


def get_sci_notation(cell):
    if cell == 0:
        return "-"

    base, expo = ("%.2e" % cell).split("e")
    return f"{base} \\times 10 ^ {int(expo)}"


def shorten_repo_name(repo):
    proj = repo.split("/", 1)[1]
    return proj[:12] + "..." if len(proj) > 15 else proj

        
def get_project_names(subjects):
    return [shorten_repo_name(repo) for repo in subjects]