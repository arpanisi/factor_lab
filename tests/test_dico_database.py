import pandas as pd

from src.rft import (
    DatabaseSelectionConfig,
    DiCoRewardConfig,
    DiscoveryResult,
    MinedFactorDatabase,
    compute_dico_reward,
    run_discovery_for_task,
)
from src.seeds import EvaluationWindow, FactorScenario, SeedCandidate, build_task_bank


def _candidate(expr="ts_mean(crypto.returns(5))", signature="sig-a", score=0.4):
    return DiscoveryResult(
        expr=expr,
        valid=True,
        signature=signature,
        score=score,
        metrics={"score": score, "rank_ic_mean": score, "valid_times": 10.0},
    )


def _crypto_frame(close):
    close = pd.Series(close, index=pd.date_range("2024-01-01", periods=len(close), freq="D"), dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
            "returns": close.pct_change(),
        }
    )


def _task():
    scenario = FactorScenario.from_benchmark("single_asset_direction", market="crypto", horizon=1)
    seed = SeedCandidate("sign(diff(crypto.close(2)))", 0.5, "seed-sig")
    return build_task_bank([seed], scenario, [EvaluationWindow("2024-01-01", "2024-01-07")])[0]


def test_dico_reward_adds_diversity_and_complementarity_for_new_factor():
    archive = MinedFactorDatabase()
    reward = compute_dico_reward(
        _candidate(score=0.4),
        archive,
        config=DiCoRewardConfig(diversity_weight=0.1, complementarity_weight=0.1),
    )

    assert reward.valid
    assert reward.reward == 0.6
    assert reward.diversity == 1.0
    assert reward.complementarity == 1.0


def test_dico_reward_penalizes_duplicate_signature():
    archive = MinedFactorDatabase()
    candidate = _candidate()
    reward = compute_dico_reward(candidate, archive)
    assert archive.maybe_add(candidate, reward, task_payload={}, config=DatabaseSelectionConfig(min_score=-1.0))

    duplicate_reward = compute_dico_reward(candidate, archive)

    assert duplicate_reward.duplicate
    assert duplicate_reward.diversity == 0.0
    assert duplicate_reward.reward < candidate.score


def test_mined_factor_database_persists_jsonl(tmp_path):
    archive = MinedFactorDatabase()
    candidate = _candidate()
    reward = compute_dico_reward(candidate, archive)
    archive.maybe_add(candidate, reward, task_payload={"task": "demo"}, config=DatabaseSelectionConfig(min_score=-1.0))

    path = tmp_path / "factors.jsonl"
    archive.save_jsonl(path)
    loaded = MinedFactorDatabase.load_jsonl(path)

    assert len(loaded.records) == 1
    assert loaded.records[0].expr == candidate.expr
    assert loaded.has_signature(candidate.signature)


def test_discovery_loop_can_reward_and_store_candidates():
    archive = MinedFactorDatabase()
    task = _task()
    frame = _crypto_frame([10, 11, 12, 11, 10, 12, 13])

    def generator(task, namespace, count):
        return ("sign(diff(crypto.close(2)))",)

    results = run_discovery_for_task(
        task,
        namespace="crypto",
        data=frame,
        price_col="close",
        count=1,
        min_history=2,
        generator=generator,
        archive=archive,
        selection_config=DatabaseSelectionConfig(min_score=-1.0, min_reward=-1.0),
    )

    assert len(results) == 1
    assert results[0].reward is not None
    assert results[0].stored
    assert len(archive.records) == 1
