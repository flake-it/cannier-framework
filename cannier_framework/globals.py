import os
import pickle
import numpy as np

from argparse import Namespace


class GlobalData:
    def setup(self, n_proj):
        self.items = np.load(ITEMS_FILE)

        if n_proj != np.unique(self.items[:, 0]).shape[0]:
            print("cannier-framework: project count mismatch.")
            sys.exit(1)
    
        n_items = self.items.shape[0]
        self.features = np.load(FEATURES_FILE)
        self.features_mean = self.features.mean(axis=1)
        self.project_masks = np.zeros((n_proj, n_items), dtype=bool)
        self.project_masks[self.items[:, 0], np.arange(n_items)] = True
        self.item_times = self.features_mean[:, 2]
        self.project_times = self.project_masks @ self.item_times
        self.item_counts = self.project_masks.sum(axis=1)

        with open(DEPENDENCIES_FILE, "rb") as f:
            self.dependencies = pickle.load(f)

        for i, deps in enumerate(self.dependencies):
            self.dependencies[i] = np.unpackbits(
                deps, axis=0, count=self.item_counts[i]
            )

        self.total_pairs = np.array(
            [deps.sum() for deps in self.dependencies], dtype=float
        )


ARGS = Namespace()
DATA = GlobalData()

ITEMS_FILE = "items.npy"
FEATURES_FILE = "features.npy"
DEPENDENCIES_FILE = "dependencies.pkl"

VENV_DIR = "venv"
SHAP_DIR = "shap"
PLOTS_DIR = "plots"
PREDS_DIR = "preds"
POINTS_DIR = "points"
STDOUT_DIR = "stdout"
TABLES_DIR = "tables"
VOLUME_DIR = "volume"
CONT_HOME_DIR = os.path.join("/", "home", "user")
CONT_VOLUME_DIR = os.path.join(CONT_HOME_DIR, VOLUME_DIR)
CONT_SUBJECTS_DIR = os.path.join(CONT_HOME_DIR, "subjects")
