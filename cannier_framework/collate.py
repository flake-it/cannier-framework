import os
import json
import pickle
import sqlite3
import numpy as np

from cannier_framework.utils import (
    fetch_counters, get_counters_max, load_subjects
)

from cannier_framework.globals import (
    ARGS, VOLUME_DIR, ITEMS_FILE, FEATURES_FILE, DEPENDENCIES_FILE
)


def collate_data_proj(cur, proj_id):
    if fetch_counters(cur) != get_counters_max():
        print(f"cannier-framework: mode counter mismatch.")
        sys.exit(1)

    id_map = {}
    items = []

    cur.execute(
        "select id, n_fail_baseline, n_fail_shuffle "
        "from item "
        "where n_runs_features = ? and "
        "n_runs_baseline = ? and "
        "n_runs_shuffle = ?",
        (ARGS.n_repeats, ARGS.n_reruns, ARGS.n_reruns)
    )

    for item_id, n_fail_baseline, n_fail_shuffle in cur.fetchall():
        id_map[item_id] = len(id_map)
        data = [proj_id, n_fail_baseline, n_fail_shuffle, 0, 0, 0, 0]
        items.append(data)
        if 0 < n_fail_baseline < ARGS.n_reruns: data[3] = 1
        elif n_fail_baseline != n_fail_shuffle: data[4] = 1
        if n_fail_baseline < ARGS.n_reruns and n_fail_shuffle > 0: data[5] = 1

    repeat_counter = {}
    items = np.array(items, dtype=int)
    features = np.empty((items.shape[0], ARGS.n_repeats, 18), dtype=float)
    cur.execute("select * from features")

    for item_id, *feats in cur.fetchall():
        try: item_id = id_map[item_id]
        except KeyError: continue
        n = repeat_counter.get(item_id, 0)
        repeat_counter[item_id] = n + 1
        features[item_id, n] = feats

    if any(n != ARGS.n_repeats for n in repeat_counter.values()):
        print(f"cannier-framework: feature count mismatch.")
        sys.exit(1)

    dependencies = np.zeros((items.shape[0], items.shape[0]), dtype=int)
    cur.execute("select victim_id, polluter_id from dependency")

    for victim_id, polluter_id in cur.fetchall():
        polluter_id = id_map[polluter_id]
        items[polluter_id][6] = 1
        dependencies[id_map[victim_id], polluter_id] = 1

    return id_map, items, features, np.packbits(dependencies, axis=0)


def collate_data():
    items = []
    features = []
    dependencies = []

    for proj_id, repo in enumerate(load_subjects()):
        proj = repo.split("/", 1)[1]
        db_file = os.path.join(VOLUME_DIR, f"{proj}.sqlite3")

        with sqlite3.connect(db_file) as con:
            data_proj = collate_data_proj(con.cursor(), proj_id)

        items.extend(data_proj[1])
        features.extend(data_proj[2])
        dependencies.append(data_proj[3])

    np.save(ITEMS_FILE, np.array(items, dtype=int))
    np.save(FEATURES_FILE, np.array(features, dtype=float))
    with open(DEPENDENCIES_FILE, "wb") as f: pickle.dump(dependencies, f)
