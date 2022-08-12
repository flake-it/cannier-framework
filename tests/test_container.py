import random
import sqlite3

from cannier_framework.utils import get_counters_max
from cannier_framework.container import fetch_victims, iter_containers_proj


def test_fetch_victims(args, db_file):
    args.n_repeats = 10
    args.n_reruns = 100

    items = [
        ("test_a", 10, 100, 0,   100,  0, 0),
        ("test_b", 1,  100, 0,   100,  0, 0),
        ("test_c", 10,  99, 0,   100,  0, 0),
        ("test_d", 10, 100, 0,     1,  0, 0),
        ("test_e", 10, 100, 0,   100,  1, 1),
        ("test_f", 10, 100, 0,   100, 99, 0),
        ("test_g", 10, 100, 100, 100,  1, 0),
    ]

    random.shuffle(items)

    with sqlite3.connect(db_file) as con:
        con.cursor().executemany(
            "insert into item "
            "values (null, ?, ?, ?, ?, ?, ?, ?)",
            items
        )

    with sqlite3.connect(db_file) as con:
        assert set(fetch_victims(con.cursor())) == {"test_f", "test_g"}


def test_iter_containers_proj(args, db_file):
    items = [
        ("test_a", 5, 10, 0, 10, 0, 0),
        ("test_b", 5, 10, 0, 10, 1, 1),
        ("test_c", 5, 10, 0, 10, 1, 0),
    ]

    random.shuffle(items)

    with sqlite3.connect(db_file) as con:
        cur = con.cursor()

        cur.execute(
            "replace into counters "
            "values (1, 0, 2, 5, 8)"
        )

        cur.executemany(
            "insert into item "
            "values (null, ?, ?, ?, ?, ?, ?, ?)",
            items
        )

    args.n_repeats = 5
    args.n_reruns = 10

    containers = {
        cont.container_id: cont for cont in iter_containers_proj(
            {"churn", "features", "baseline", "victim"}, db_file, "proj", (), 
            get_counters_max()
        )
    }

    assert set(containers) == {
        "proj_baseline_0", "proj_baseline_1", "proj_baseline_2", 
        "proj_baseline_3", "proj_baseline_4", "proj_victim_0", "proj_churn_0", 
        "proj_features_0", "proj_features_1", "proj_features_2"
    }

    assert containers["proj_victim_0"].victim_nodeid == "test_c"