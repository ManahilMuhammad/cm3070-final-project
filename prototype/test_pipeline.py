import io
import json
import pytest
import instrumentation as inst
import pipeline

@pytest.fixture(autouse=True)
def clean_records():
    inst.reset()
    yield
    inst.reset()


def chat_response(content):
    return {"message": {"content": content}}


# --> fuse()

def test_fuse_combines_sections():
    result = pipeline.fuse("t", "s", "n", "f")
    assert result == "[TRANSCRIPT]\nt\n\n[SLIDE]\ns\n\n[NOTES]\nn\n\n[FIGURE]\nf"


def test_fuse_skips_empty():
    result = pipeline.fuse("t", "", "  ", "None")
    assert result == "[TRANSCRIPT]\nt"


def test_fuse_is_timed_by_instrumentation():
    pipeline.fuse("t", "", "", "")
    assert "fusion of extracted text" in inst.aggregate()


# --> extract_slides()

def test_extract_slides_none():
    assert pipeline.extract_slides(None) == ""


def test_extract_slides_unsupported():
    fake = io.BytesIO(b"data")
    fake.name = "notes.txt"
    assert pipeline.extract_slides(fake) == ""


def test_extract_slides_reads_pdf_text(tmp_path):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello world from pdf")
    data = doc.tobytes()
    doc.close()

    fake = io.BytesIO(data)
    fake.name = "slides.pdf"

    text = pipeline.extract_slides(fake)
    assert "hello world from pdf" in text


def test_extract_slides_reads_pptx_text():
    pptx = pytest.importorskip("pptx")
    from pptx.util import Inches

    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    box.text_frame.text = "hello world from pptx"

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    buf.name = "slides.pptx"

    text = pipeline.extract_slides(buf)
    assert "hello world from pptx" in text


# --> describe_figure()

def test_describe_figure_none():
    assert pipeline.describe_figure(None) == ""


def test_describe_figure_calls_ollama_chat(monkeypatch):
    captured = {}

    def fake_chat(model, messages):
        captured["model"] = model
        captured["images"] = messages[0]["images"]
        return chat_response("a diagram of X")

    monkeypatch.setattr(pipeline.ollama, "chat", fake_chat)

    fake = io.BytesIO(b"imgbytes")
    result = pipeline.describe_figure(fake)

    assert result == "a diagram of X"
    assert captured["model"] == "qwen2.5vl:latest"
    assert captured["images"] == [b"imgbytes"]
    assert "figure description" in inst.aggregate()


# --> make_summary()

def test_make_summary_calls_ollama_chat(monkeypatch):
    monkeypatch.setattr(
        pipeline.ollama, "chat", lambda model, messages: chat_response("# Summary\n...")
    )

    result = pipeline.make_summary("some combined text")

    assert result == "# Summary\n..."
    assert "summarisation" in inst.aggregate()


# --> generate_ques() / make_quiz()

def test_generate_ques_succeeds_first_try(monkeypatch):
    valid = json.dumps({"question": "2+2?", "answer": "4", "options": ["3", "4", "5", "6"]})
    monkeypatch.setattr(pipeline.ollama, "chat", lambda **kw: chat_response(valid))

    spec = {
        "type": "multiple-choice",
        "shape": {},
        "instructions": "",
    }
    q = pipeline.generate_ques("content", spec)

    assert q["type"] == "multiple-choice"
    assert q["question"] == "2+2?"
    assert q["answer"] == "4"
    assert "failed" not in q


def test_generate_ques_retries_on_bad_json(monkeypatch):
    calls = {"n": 0}
    valid = json.dumps({"question": "cap of France?", "answer": "Paris", "options": None})

    def fake_chat(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return chat_response("not json")
        return chat_response(valid)

    monkeypatch.setattr(pipeline.ollama, "chat", fake_chat)

    spec = {"type": "short-answer", "shape": {}, "instructions": ""}
    q = pipeline.generate_ques("content", spec)

    assert calls["n"] == 2
    assert q["answer"] == "Paris"


def test_generate_ques_blank_answer_leak(monkeypatch):
    # answer word appears in the question -> invalid -> exhausts retries -> failed
    bad = json.dumps({"question": "Paris is the _____ of France", "answer": "Paris", "options": None})
    monkeypatch.setattr(pipeline.ollama, "chat", lambda **kw: chat_response(bad))

    spec = {"type": "fill-in-the-blank", "shape": {}, "instructions": ""}
    q = pipeline.generate_ques("content", spec, retries=2)

    assert q["failed"] is True
    assert q["answer"] is None


def test_make_quiz_filters_failed(monkeypatch):
    def fake_generate_ques(content, spec, retries=3):
        if spec["type"] == "true-false":
            return {"type": "true-false", "question": "q", "answer": None, "options": None, "failed": True}
        return {"type": spec["type"], "question": "q", "answer": "a", "options": None}

    monkeypatch.setattr(pipeline, "generate_ques", fake_generate_ques)

    quiz = pipeline.make_quiz("combined text")

    assert len(quiz) == 3
    assert all(q["type"] != "true-false" for q in quiz)
    assert "create quiz" in inst.aggregate()


# --> extract_topics()

def test_extract_topics_dedupes(monkeypatch):
    raw = json.dumps({"topics": ["Neural Nets", "neural nets", "Backprop"]})
    monkeypatch.setattr(pipeline.ollama, "generate", lambda **kw: {"response": raw})

    topics = pipeline.extract_topics("lecture text", n=8)

    assert topics == ["Neural Nets", "Backprop"]


def test_extract_topics_bad_json_fallback(monkeypatch):
    monkeypatch.setattr(pipeline.ollama, "generate", lambda **kw: {"response": "not json"})

    topics = pipeline.extract_topics("lecture text")

    assert topics == ["General"]


# --> tag_topic()

def test_tag_topic_no_topics():
    assert pipeline.tag_topic("some question", []) == "General"


def test_tag_topic_resolves_numeric_index(monkeypatch):
    monkeypatch.setattr(pipeline.ollama, "generate", lambda **kw: {"response": '{"topic": "2"}'})
    topics = ["Alpha", "Beta", "Gamma"]

    assert pipeline.tag_topic("q", topics) == "Beta"


def test_tag_topic_index_out_of_range(monkeypatch):
    monkeypatch.setattr(pipeline.ollama, "generate", lambda **kw: {"response": '{"topic": "9"}'})

    assert pipeline.tag_topic("q", ["Alpha", "Beta"]) == "Unclassified"


def test_tag_topic_exact_text(monkeypatch):
    monkeypatch.setattr(pipeline.ollama, "generate", lambda **kw: {"response": '{"topic": "Gamma"}'})

    assert pipeline.tag_topic("q", ["Alpha", "Gamma"]) == "Gamma"


def test_tag_topic_bad_json(monkeypatch):
    monkeypatch.setattr(pipeline.ollama, "generate", lambda **kw: {"response": "not json"})

    assert pipeline.tag_topic("q", ["Alpha"]) == "Unclassified"


# --> generate_score()

def test_generate_score_correctness():
    results = [
        {"time": 4.0, "num_words": 4, "user_answer": "Paris", "correct_answer": "paris", "topic": "geo"},
        {"time": 2.0, "num_words": 4, "user_answer": "wrong", "correct_answer": "right", "topic": "geo"},
    ]

    out = json.loads(pipeline.generate_score(results))

    assert results[0]["correct"] is True
    assert results[1]["correct"] is False
    assert results[1]["confidence"] == 0.0
    assert 0.0 <= results[0]["confidence"] <= 1.0

    assert len(out) == 1
    assert out[0]["topic"] == "geo"
    assert out[0]["score"] == "1/2"
    assert "generate score" in inst.aggregate()


def test_generate_score_zero_stdev():
    results = [{"time": 1.0, "num_words": 2, "user_answer": "a", "correct_answer": "a", "topic": "t"}]

    pipeline.generate_score(results)

    assert results[0]["correct"] is True
    assert results[0]["confidence"] == pytest.approx(0.5)


def test_generate_score_sorts_topics():
    results = [
        {"time": 1.0, "num_words": 2, "user_answer": "a", "correct_answer": "a", "topic": "strong"},
        {"time": 1.0, "num_words": 2, "user_answer": "x", "correct_answer": "y", "topic": "weak"},
    ]

    out = json.loads(pipeline.generate_score(results))

    assert [o["topic"] for o in out] == ["weak", "strong"]


# --> generate_feedback() / create_plan()

def test_generate_feedback_calls_ollama_chat(monkeypatch):
    monkeypatch.setattr(pipeline.ollama, "chat", lambda model, messages: chat_response("Great job!"))

    result = pipeline.generate_feedback("[]", "combined")

    assert result == "Great job!"
    assert "generate feedback" in inst.aggregate()


def test_create_plan_calls_ollama_chat(monkeypatch):
    monkeypatch.setattr(pipeline.ollama, "chat", lambda model, messages: chat_response("Study X first."))

    result = pipeline.create_plan("[]", "combined")

    assert result == "Study X first."
    assert "create plan" in inst.aggregate()


# --> unload_all()

def test_unload_all_clears_models():
    pipeline._models["dummy"] = object()

    pipeline.unload_all()

    assert pipeline._models == {}


# --> release_llm() / end_of_phase wiring into instrumentation

def test_release_llm_error_skips_inst(monkeypatch):
    def fake_generate(model, prompt, keep_alive):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(pipeline.ollama, "generate", fake_generate)

    calls = []
    monkeypatch.setattr(pipeline.inst, "report", lambda *a, **k: calls.append("report"))
    monkeypatch.setattr(pipeline.inst, "save_run", lambda *a, **k: calls.append("save_run"))
    monkeypatch.setattr(pipeline.inst, "reset", lambda *a, **k: calls.append("reset"))

    ok = pipeline.release_llm()

    assert ok is False
    assert calls == []


def test_release_llm_end_of_phase_true(monkeypatch):
    monkeypatch.setattr(pipeline.ollama, "generate", lambda model, prompt, keep_alive: {})

    calls = []
    monkeypatch.setattr(pipeline.inst, "report", lambda *a, **k: calls.append(("report", a, k)))
    monkeypatch.setattr(pipeline.inst, "save_run", lambda *a, **k: calls.append(("save_run", a, k)))
    monkeypatch.setattr(pipeline.inst, "reset", lambda *a, **k: calls.append(("reset", a, k)))

    ok = pipeline.release_llm(label="my-run", end_of_phase=True)

    assert ok is True
    assert [c[0] for c in calls] == ["report", "save_run", "reset"]
    assert calls[1][1] == ("my-run",)


def test_release_llm_end_of_phase_false(monkeypatch):
    monkeypatch.setattr(pipeline.ollama, "generate", lambda model, prompt, keep_alive: {})

    calls = []
    monkeypatch.setattr(pipeline.inst, "report", lambda *a, **k: calls.append("report"))
    monkeypatch.setattr(pipeline.inst, "save_run", lambda *a, **k: calls.append("save_run"))
    monkeypatch.setattr(pipeline.inst, "reset", lambda *a, **k: calls.append("reset"))

    ok = pipeline.release_llm(model="qwen2.5vl:latest", end_of_phase=False)

    assert ok is True
    assert calls == []


def test_release_llm_integration(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.ollama, "generate", lambda model, prompt, keep_alive: {})

    log_path = tmp_path / "timings.jsonl"
    real_save_run = inst.save_run
    monkeypatch.setattr(
        pipeline.inst, "save_run",
        lambda label, notes="", path=log_path: real_save_run(label, notes, path)
    )

    pipeline.fuse("transcript text", "", "", "")
    assert "fusion of extracted text" in inst.aggregate()

    ok = pipeline.release_llm(label="integration-run", end_of_phase=True)

    assert ok is True
    assert inst._records == []

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["label"] == "integration-run"
    assert "fusion of extracted text" in record["stages"]
