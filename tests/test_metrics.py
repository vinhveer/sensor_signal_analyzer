import numpy as np

from sensorcls.engine.metrics import build_scores


def test_build_scores_keys():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    scores = build_scores(y_true, y_pred)
    assert "accuracy" in scores
    assert "precision" in scores
    assert "recall" in scores
    assert "f1score" in scores
    assert "confusion" in scores
