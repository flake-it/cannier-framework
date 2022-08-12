import pytest
import numpy as np

from cannier_framework.model import Model


PROJECT_TIMES = np.array([10, 12])
ITEM_COSTS = np.array([2, 3, 5, 2, 3, 7])
LABELS = np.array([True, False, True, False, True, False])
PREDS = np.array([[[.9, .1, .2, .8, .7, .3], [.4, .2, .7, .2, .4, .6],]])

PROJECT_MASKS = np.array([
    [True, False, True, False, True, False],
    [False, True, False, True, False, True],
])


def test_get_confusion():
    model = Model(None, None, None)
    model.labels = LABELS
    model.preds = PREDS[:, (0,)]
    categories, amb_mask = model.get_confusion(.3, .8, 1, 0)

    categories_expected = np.array([
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])

    assert (categories == categories_expected).all()
    amb_mask_expected = np.array([False, False, False, False, True, True])
    assert (amb_mask == amb_mask_expected).all()


def test_get_scores_simple(args, data):
    model = Model(None, None, None)
    model.labels = LABELS
    model.preds = PREDS
    args.n_repeats = 2
    data.project_times = PROJECT_TIMES
    perf, cost = model.get_scores_simple(ITEM_COSTS, .3, .8, 1)
    assert perf == pytest.approx(2/3)
    assert cost == 35.5


def test_get_scores_simple_no_model(data):
    model = Model(None, None, None)
    data.project_times = PROJECT_TIMES
    perf, cost = model.get_scores_simple(ITEM_COSTS, .3, .8, 0)
    assert perf == 1
    assert cost == 22


def test_get_scores_full(args, data):
    model = Model(None, None, None)
    model.labels = LABELS
    model.preds = PREDS
    model.project_masks = PROJECT_MASKS
    args.n_repeats = 2
    data.project_times = PROJECT_TIMES
    scores = model.get_scores_full(ITEM_COSTS, .3, .8, 1)
    scores = np.ma.masked_where(np.isnan(scores), scores)
    
    scores_expected = np.array([
        [ .0, .5, .0, 2.5, np.nan, 16.5],
        [2.5, .0, .5,  .0, np.nan, 19.0],
        [2.5, .5, .5, 2.5,    2/3, 35.5]
    ])

    scores_expected = np.ma.masked_where(
        np.isnan(scores_expected), scores_expected
    )

    assert np.isclose(scores, scores_expected).all()


def test_get_scores_full_no_model(data):
    model = Model(None, None, None)
    model.labels = LABELS
    model.project_masks = PROJECT_MASKS
    data.project_times = PROJECT_TIMES
    scores = model.get_scores_full(ITEM_COSTS, .3, .8, 0)

    scores_expected = np.array([
        [3, 0, 0, 0, 1, 10],
        [0, 0, 0, 3, 1, 12],
        [3, 0, 0, 3, 1, 22]
    ])

    assert (scores == scores_expected).all()