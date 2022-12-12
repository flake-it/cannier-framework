import os
import pickle
import numpy as np

from argparse import Namespace


ITEMS_FILE = "items.npy"
FEATURES_FILE = "features.npy"
DEPENDENCIES_FILE = "dependencies.pkl"


class GlobalData:
    def __init__(self):
        self.features_mean = None
        self.project_masks = None
        self.items_cache = {}
        self.features_cache = {}
        self.labels_cache = {}

    def get_items_features(self, n_proj):
        self.items = np.load(ITEMS_FILE)

        if n_proj != np.unique(self.items[:, 0]).shape[0]:
            print("cannier-framework: project count mismatch.")
            sys.exit(1)

        self.features = np.load(FEATURES_FILE)

    def get_features_mean(self):
        self.features_mean = self.features.mean(axis=1)

    def get_rest(self, n_proj):
        n_items = self.items.shape[0]
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

    def get_model_data(self, domain_idx, labels_idx, features_mask):
        if domain_idx is not None: 
            domain_mask = self.items[:, domain_idx].astype(bool)

        try: 
            items, project_masks = self.items_cache[domain_idx]
        except KeyError:
            items = self.items
            project_masks = self.project_masks

            if domain_idx is not None:
                items = items[domain_mask]

                if project_masks is not None: 
                    project_masks = project_masks[:, domain_mask]

                self.items_cache[domain_idx] = items, project_masks

        try:
            key = domain_idx, features_mask
            features, features_mean = self.features_cache[key]
        except KeyError:
            features = self.features
            features_mean = self.features_mean

            if domain_idx is not None:
                features = features[domain_mask]

                if features_mean is not None: 
                    features_mean = features_mean[domain_mask]

            if features_mask is not None:
                features = features[:, :, features_mask]
                
                if features_mean is not None: 
                    features_mean = features_mean[:, features_mask]

            if domain_idx is not None or features_mask is not None:
                self.features_cache[key] = features, features_mean

        try:
            labels = self.labels_cache[(domain_idx, labels_idx)]
        except KeyError:
            labels = self.items[:, labels_idx].astype(bool)
            if domain_idx is not None: labels = labels[domain_mask]
            self.labels_cache[(domain_idx, labels_idx)] = labels

        return items, features, features_mean, project_masks, labels


ARGS = Namespace()
DATA = GlobalData()
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
