import numpy as np

from src.utils.metrics import compute_cmai_metrics, find_optimal_thresholds


def test_threshold_finder_returns_one_threshold_per_class():
    y_true = np.zeros((4, 29), dtype=int)
    y_score = np.zeros((4, 29), dtype=float)
    y_true[:, 0] = [1, 0, 1, 0]
    y_score[:, 0] = [0.8, 0.2, 0.7, 0.1]
    thresholds = find_optimal_thresholds(y_true, y_score)
    assert thresholds.shape == (29,)


def test_compute_cmai_metrics_has_summary_keys():
    y_true = np.zeros((3, 29), dtype=int)
    y_pred = np.zeros((3, 29), dtype=int)
    y_score = np.zeros((3, 29), dtype=float)
    y_true[0, 0] = 1
    y_pred[0, 0] = 1
    y_score[0, 0] = 0.9
    metrics = compute_cmai_metrics(y_true, y_pred, y_score)
    assert "micro_f1" in metrics
    assert "macro_f1" in metrics

