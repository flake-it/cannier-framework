import pytest
import numpy as np

from cannier_framework.technique import get_pareto_front, get_exp_reruns


def test_get_pareto_front():
    points = np.random.rand(100000, 2)
    
    for i in get_pareto_front(points):
        assert (points[points[:, 0] <= points[i, 0], 1] >= points[i, 1]).all()
        assert (points[points[:, 1] <= points[i, 1], 0] >= points[i, 0]).all()


@pytest.mark.parametrize(
    "p_fail,n_reruns", 
    [
        (.05, 10), (.05, 20), (.05, 50),
        (.02, 10), (.02, 20), (.02, 50),
        (.01, 10), (.01, 20), (.01, 50)
    ]
)
def test_get_exp_reruns(p_fail, n_reruns):
    observed = get_exp_reruns(p_fail, n_reruns)
    p = p_fail, 1 - p_fail
    runs_first = np.random.choice(2, 100000, p=p)
    runs_rest = np.random.choice(2, (n_reruns - 1, 100000), p=p)
    terminate = runs_first != runs_rest
    terminate[-1] = True
    expected = terminate.argmax(axis=0).mean() + 2
    assert observed == pytest.approx(expected, abs=1e-0)