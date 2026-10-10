from datetime import datetime

from declutter import RunResult, next_run
from state import State


def test_state_lifecycle(tmp_path):
    s = State(str(tmp_path / "s.db"))
    s.set("a", "none")
    s.set("b", "proposed", "memes", "meme=0.9")
    s.fail("c", "timeout")
    s.commit()
    assert s.known(max_failures=3) == {"a", "b"}
    assert s.retry_ids(3) == ["c"]
    assert s.proposed() == {"b": "memes"}
    s.set("b", "kept")
    for _ in range(3):
        s.fail("c", "timeout")
    s.commit()
    assert s.proposed() == {}
    assert s.retry_ids(3) == []
    assert "c" in s.known(max_failures=3)  # failed too often: do not retry forever
    assert s.counts() == {"none": 1, "kept": 1, "retry": 1}


def test_kept_keeps_category(tmp_path):
    s = State(str(tmp_path / "s.db"))
    s.set("b", "proposed", "memes", "x")
    s.set("b", "kept")
    row = s.db.execute("SELECT status, category FROM assets").fetchone()
    assert row == ("kept", "memes")


def test_next_run():
    assert next_run(datetime(2026, 10, 10, 2, 0), "03:30") == datetime(2026, 10, 10, 3, 30)
    assert next_run(datetime(2026, 10, 10, 3, 30), "03:30") == datetime(2026, 10, 11, 3, 30)


def test_summary():
    r = RunResult(scanned=10, new={"memes": 2, "tiktok": 0}, kept=1)
    assert "2 new candidates (M: 2)" in r.summary({"memes": "M", "tiktok": "T"})


def test_thresholds_from_env(monkeypatch):
    from declutter import Config
    monkeypatch.setenv("IMMICH_URL", "http://immich:2283")
    monkeypatch.setenv("IMMICH_API_KEY", "x")
    monkeypatch.setenv("ML_URL", "http://ml:3003")
    monkeypatch.setenv("THRESHOLD_MEME", "0.7")
    monkeypatch.setenv("CATEGORIES", "memes,receipts")
    cfg = Config.from_env()
    assert cfg.thresholds.meme == 0.7 and cfg.thresholds.screen == 0.6
    assert cfg.categories == {"memes", "receipts"}
