import random
import sqlite3
import numpy as np

from cannier_framework.collate import collate_data_proj


def test_collate_data_proj_items(args, db_file):
    items_input = [
        ("test_a", 0, 10, 0,  10,  0, 0),
        ("test_b", 0, 10, 5,  10,  0, 1),
        ("test_c", 0, 10, 0,  10,  5, 0),
        ("test_d", 0,  9, 0,  10,  0, 1),
        ("test_e", 0, 10, 5,  10,  5, 0),
        ("test_f", 0, 10, 0,  10, 10, 1),
        ("test_g", 0, 10, 10, 10,  5, 0)
    ]

    random.shuffle(items_input)
    args.n_repeats = 0
    args.n_reruns = 10

    with sqlite3.connect(db_file) as con:
        cur = con.cursor()

        cur.execute(
            "replace into counters "
            "values (1, 1, 0, 10, 10)"
        )

        cur.executemany(
            "insert into item "
            "values (null, ?, ?, ?, ?, ?, ?, ?)",
            items_input
        )

        cur.execute(
            "select id, nodeid "
            "from item"
        )

        id_to_nodeid = {
            item_id: nodeid for item_id, nodeid in cur.fetchall()
        }

        id_map, items_output, *_ = collate_data_proj(cur, 0)

    nodeid_to_id = {id_to_nodeid[id1]: id2 for id1, id2 in id_map.items()}
    assert "test_d" not in nodeid_to_id

    order = [
        nodeid_to_id[nodeid] for nodeid in (
            "test_a", "test_b", "test_c", "test_e", "test_f", "test_g"
        )
    ]

    items_expected = np.array([
        [0, 0,  0, 0, 0, 0, 0],
        [0, 5,  0, 1, 0, 0, 0],
        [0, 0,  5, 0, 1, 1, 0],
        [0, 5,  5, 1, 0, 1, 0],
        [0, 0, 10, 0, 1, 1, 0],
        [0, 10, 5, 0, 1, 0, 0],
    ])

    assert (items_output[order] == items_expected).all()


def test_collate_data_proj_features(args, db_file):
    items = [
        ("test_a", 5, 10,  0, 10,  0, 0),
        ("test_b", 5, 10,  0, 10, 10, 1),
        ("test_c", 5,  9,  0, 10,  0, 1),
        ("test_d", 5, 10, 10, 10,  0, 0),
    ]

    features_input = [[i + 1] * 19 for i in range(len(items))] * 5
    random.shuffle(features_input)
    args.n_repeats = 5
    args.n_reruns = 10

    with sqlite3.connect(db_file) as con:
        cur = con.cursor()

        cur.execute(
            "replace into counters "
            "values (1, 1, 5, 10, 10)"
        )

        cur.executemany(
            "insert into item "
            "values (null, ?, ?, ?, ?, ?, ?, ?)",
            items
        )

        cur.executemany(
            "insert into features "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
            features_input
        )

        _, _, features_output, _ = collate_data_proj(cur, 0)

    features_expected = np.array(
        [item_id * np.ones((5, 18)) for item_id in (1, 2, 4)]
    )

    assert (features_output == features_expected).all()


def test_collate_data_proj_dependencies(args, db_file):
    items = [
        ("test_a", 0, 5, 0, 5, 0, 1),
        ("test_b", 0, 5, 0, 5, 5, 1),
        ("test_c", 0, 5, 5, 5, 0, 1),
        ("test_d", 0, 5, 0, 5, 2, 1),
    ]

    dependencies_input = [(2, 1), (3, 1), (4, 2)]
    random.shuffle(dependencies_input)
    args.n_repeats = 0
    args.n_reruns = 5

    with sqlite3.connect(db_file) as con:
        cur = con.cursor()

        cur.execute(
            "replace into counters "
            "values (1, 1, 0, 5, 5)"
        )

        cur.executemany(
            "insert into item "
            "values (null, ?, ?, ?, ?, ?, ?, ?)",
            items
        )

        cur.executemany(
            "insert into dependency "
            "values (?, ?)",
            dependencies_input
        )

        _, _, _, dependencies_output = collate_data_proj(con.cursor(), 0)

    dependencies_output = np.unpackbits(dependencies_output, axis=0, count=4)

    dependencies_expected = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    assert (dependencies_output == dependencies_expected).all()