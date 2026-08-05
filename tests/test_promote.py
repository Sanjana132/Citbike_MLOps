"""Champion/challenger promotion rule.

The decision function is deliberately pure, so the rule that governs what
reaches production can be tested without an MLflow server.
"""

from __future__ import annotations

import pytest

from src.models.promote import (
    Candidate,
    decide_promotion,
    relative_improvement,
    select_best_candidate,
)


def make(name: str, mae: float, rmse: float = 1.0) -> Candidate:
    return Candidate(name=name, run_id=f"run-{name}", metrics={"mae": mae, "rmse": rmse}, model_uri=f"runs:/{name}/model")


def test_first_model_is_promoted_unconditionally():
    """Something has to serve; with no champion the best candidate wins."""
    decision = decide_promotion([make("lightgbm", 5.0)], champion_score=None)
    assert decision.promote
    assert decision.challenger.name == "lightgbm"
    assert "No existing production model" in decision.reason


def test_clear_improvement_is_promoted():
    decision = decide_promotion(
        [make("lightgbm", 8.0)], champion_score=10.0, min_relative_improvement=0.02
    )
    assert decision.promote
    assert decision.relative_improvement == pytest.approx(0.2)


def test_marginal_improvement_is_rejected():
    """1% better is noise, not progress - the champion stays."""
    decision = decide_promotion(
        [make("lightgbm", 9.9)], champion_score=10.0, min_relative_improvement=0.02
    )
    assert not decision.promote
    assert "does not clear" in decision.reason


def test_improvement_exactly_at_the_bar_is_rejected():
    """The margin is strict: ties go to the incumbent."""
    decision = decide_promotion(
        [make("lightgbm", 9.8)], champion_score=10.0, min_relative_improvement=0.02
    )
    assert not decision.promote


def test_worse_challenger_is_rejected():
    decision = decide_promotion([make("lightgbm", 12.0)], champion_score=10.0)
    assert not decision.promote
    assert decision.relative_improvement < 0


def test_promotion_is_model_agnostic():
    """Whichever family wins on the holdout goes to production."""
    decision = decide_promotion(
        [make("lightgbm", 9.0), make("deep", 6.0)],
        champion_score=10.0,
        min_relative_improvement=0.02,
    )
    assert decision.promote
    assert decision.challenger.name == "deep"

    decision = decide_promotion(
        [make("lightgbm", 6.0), make("deep", 9.0)],
        champion_score=10.0,
        min_relative_improvement=0.02,
    )
    assert decision.challenger.name == "lightgbm"


def test_best_candidate_respects_metric_direction():
    candidates = [make("a", 5.0), make("b", 3.0)]
    assert select_best_candidate(candidates, "mae").name == "b"

    # r2 is higher-is-better, so the ordering must flip.
    higher = [
        Candidate("a", "r1", {"r2": 0.4}, "uri-a"),
        Candidate("b", "r2", {"r2": 0.8}, "uri-b"),
    ]
    assert select_best_candidate(higher, "r2").name == "b"


def test_no_candidates_never_promotes():
    decision = decide_promotion([], champion_score=10.0)
    assert not decision.promote
    assert "No candidates" in decision.reason


def test_zero_champion_score_does_not_divide_by_zero():
    """A perfect champion must not cause a crash or an automatic promotion."""
    decision = decide_promotion([make("lightgbm", 1.0)], champion_score=0.0)
    assert not decision.promote

    assert relative_improvement(1.0, 0.0, "mae") == 0.0 or not decision.promote


def test_missing_metric_is_an_error_not_a_silent_pass():
    """A candidate that failed to produce the promotion metric must not win."""
    broken = Candidate("broken", "run-x", {"rmse": 1.0}, "uri")
    with pytest.raises(KeyError, match="mae"):
        decide_promotion([broken], champion_score=10.0)


# --------------------------------------------------------------------------
# Promotion must compare what actually gets served
# --------------------------------------------------------------------------

def test_promotion_uses_as_served_metrics_not_in_process_ones():
    """Regression test for the worst defect this project produced.

    The LSTM was promoted on an in-process holdout MAE of 22.25, computed from
    true 168-hour sequences. Its pyfunc serving path rebuilds those sequences
    from five lag columns and actually scores 39.27 - 60% worse than the
    LightGBM champion (24.57) it displaced. Promotion shipped a worse model
    while its scoreboard said otherwise.

    Candidate.metrics must therefore hold the as-served numbers, so that a model
    whose serving path cannot reproduce its holdout loses.
    """
    lightgbm = Candidate("lightgbm", "run-lgb", {"mae": 24.566}, "uri-lgb")
    # As-served metrics for the deep model, not its flattering in-process ones.
    deep_as_served = Candidate("deep", "run-deep", {"mae": 39.267}, "uri-deep")

    decision = decide_promotion(
        [lightgbm, deep_as_served], champion_score=24.566, min_relative_improvement=0.02
    )
    assert not decision.promote
    assert decision.challenger.name == "lightgbm"

    # And the counterfactual: scored on in-process numbers it would have won.
    deep_in_process = Candidate("deep", "run-deep", {"mae": 22.245}, "uri-deep")
    bad = decide_promotion(
        [lightgbm, deep_in_process], champion_score=24.566, min_relative_improvement=0.02
    )
    assert bad.promote and bad.challenger.name == "deep"


def test_score_as_served_round_trips_through_the_serving_path(monkeypatch):
    """score_as_served must load the logged model, not reuse the in-memory one."""
    import numpy as np
    import pandas as pd

    from src.models import train as train_module

    loaded = {}

    class FakeServed:
        def predict(self, frame):
            loaded["rows"] = len(frame)
            loaded["dtype"] = str(frame["a"].dtype)
            return np.full(len(frame), 10.0)

    import mlflow.pyfunc

    monkeypatch.setattr(
        mlflow.pyfunc,
        "load_model",
        lambda uri: (loaded.setdefault("uri", uri), FakeServed())[1],
    )

    class FakeDataset:
        features = ["a"]

        def xy(self, split):
            assert split == "test", "promotion must score the holdout, not train"
            frame = pd.DataFrame({"a": [1, 2, 3]})  # int, as a real panel would be
            return frame, pd.Series([10.0, 10.0, 10.0])

    metrics = train_module.score_as_served("runs:/abc/model", FakeDataset())
    assert loaded["uri"] == "runs:/abc/model"
    assert loaded["rows"] == 3
    # Input must reach the model as float64, matching the logged signature.
    assert loaded["dtype"] == "float64"
    assert metrics["mae"] == 0.0
