import json
import time
import pytest
import instrumentation as inst

@pytest.fixture(autouse=True)
def clean_records():
    inst.reset()
    yield
    inst.reset()


# --> stage()

def test_stage_records_duration():
    with inst.stage("work"):
        time.sleep(0.01)

    assert len(inst._records) == 1
    name, dur = inst._records[0]
    assert name == "work"
    assert dur > 0


def test_stage_records_even_on_exception():
    with pytest.raises(ValueError):
        with inst.stage("boom"):
            raise ValueError("x")

    assert inst._records == [("boom", inst._records[0][1])]


# --> timed()

def test_timed_default_label():
    @inst.timed()
    def foo():
        return 42

    assert foo() == 42
    assert inst._records[0][0] == "foo"


def test_timed_custom_label():
    @inst.timed("custom")
    def foo():
        pass

    foo()
    assert inst._records[0][0] == "custom"


# --> reset()

def test_reset_clears_records():
    with inst.stage("a"):
        pass
    inst.reset()
    assert inst._records == []


# --> aggregate()

def test_aggregate_groups_calls():
    with inst.stage("a"):
        time.sleep(0.01)
    with inst.stage("a"):
        time.sleep(0.02)
    with inst.stage("b"):
        time.sleep(0.01)

    agg = inst.aggregate()

    assert set(agg) == {"a", "b"}
    assert agg["a"]["calls"] == 2
    assert agg["a"]["total"] == pytest.approx(sum(d for n, d in inst._records if n == "a"))
    assert agg["a"]["mean"] == pytest.approx(agg["a"]["total"] / 2)
    assert agg["a"]["max"] >= agg["a"]["mean"]


def test_aggregate_empty():
    assert inst.aggregate() == {}


# --> report()

def test_report_empty(capsys):
    total = inst.report()
    assert total == 0.0
    assert "No timings recorded" in capsys.readouterr().out


def test_report_prints_table(capsys):
    with inst.stage("a"):
        time.sleep(0.01)

    total = inst.report(title="MyRun")
    out = capsys.readouterr().out

    assert total > 0
    assert "MyRun" in out
    assert "a" in out
    assert "TOTAL" in out


# --> save_run()

def test_save_run_writes_jsonl(tmp_path):
    path = tmp_path / "timings.jsonl"

    with inst.stage("a"):
        time.sleep(0.01)
    inst.save_run("run1", notes="first", path=path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["label"] == "run1"
    assert record["notes"] == "first"
    assert record["stages"]["a"]["calls"] == 1


def test_save_run_nothing_to_save(tmp_path, capsys):
    path = tmp_path / "timings.jsonl"
    inst.save_run("empty", path=path)

    assert not path.exists()
    assert "Nothing to save" in capsys.readouterr().out


# --> _load()

def test_load_missing_file(tmp_path):
    assert inst._load(tmp_path / "missing.jsonl") == []


# --> compare()

def test_compare_missing_labels(tmp_path, capsys):
    path = tmp_path / "timings.jsonl"
    inst.compare("before", "after", path=path)

    assert "No saved runs labeled" in capsys.readouterr().out


def test_compare_prints_deltas(tmp_path, capsys):
    path = tmp_path / "timings.jsonl"

    with inst.stage("a"):
        time.sleep(0.01)
    inst.save_run("before", path=path)
    inst.reset()

    with inst.stage("a"):
        time.sleep(0.02)
    inst.save_run("after", path=path)

    inst.compare("before", "after", path=path)
    out = capsys.readouterr().out

    assert "before -> after" in out
    assert "a" in out
    assert "TOTAL" in out
