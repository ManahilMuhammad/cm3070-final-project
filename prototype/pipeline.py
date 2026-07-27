import ollama
import json
from faster_whisper import WhisperModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import numpy as np
import fitz
from pptx import Presentation
from collections import defaultdict
import io
import streamlit as st
import instrumentation as inst
import statistics, math, json
from collections import defaultdict
import gc
import torch

_models = {}

def _get(name, loader):
    if name not in _models:
        _models[name] = loader()
    return _models[name]

def unload_all():
    """Release the ingestion models. Call once extraction is finished."""
    _models.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_trocr():
    name = "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained(name)
    model = VisionEncoderDecoderModel.from_pretrained(name).to(DEVICE)
    model.eval()
    return processor, model

@inst.timed("transcription")
def transcribe_audio(path):
    if path is None:
        return ""

    whisper_model = _get("whisper", lambda: WhisperModel("base", compute_type="int8"))
    
    segments, info = whisper_model.transcribe(path)
    transcript = " ".join(segment.text for segment in segments)

    return transcript

@inst.timed("slide extraction")
def extract_slides(path):
    if path is None:
        return ""
    name = path.name.lower()
    if name.endswith('.pdf'):
        data = path.read()
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    elif name.endswith('.pptx'):
        prs = Presentation(path)
        lines = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = ''.join(run.text for run in para.runs)
                        if text.strip():
                            lines.append(text)
        return '\n'.join(lines)
    else:
        return ""

@inst.timed("OCR")
def ocr_notes(path):
    if path is None:
        return ""

    trocr_processor, trocr_model = _get("trocr", _load_trocr)

    min_area = 0.0004
    pad = 6

    image = Image.open(path).convert("RGB")

    img = np.array(image.convert("L"))
    h, w = img.shape
    
    mean_brightness = img.mean()

    if mean_brightness < 127:         
        thresh_type = cv2.THRESH_BINARY       
    else:                            
        thresh_type = cv2.THRESH_BINARY_INV   

    _, binary = cv2.threshold(img, 0, 255, thresh_type + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 20), 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] * b[3] > min_area * w * h]
    boxes.sort(key=lambda b: b[1])

    lines = []
    for x, y, bw, bh in boxes:
        lines.append(image.crop((max(0, x - pad), max(0, y - pad),
                                    min(w, x + bw + pad), min(h, y + bh + pad))))

    notes = ""
    for line in lines:
        pixel_values = trocr_processor(images=line.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        notes += trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0] + "\n"

    return notes

@inst.timed("figure description")
def describe_figure(path): 
    prompt = """
You are helping create study notes from lecture material.
Describe the figure, chart, or diagram in the uploaded image clearly including all key data,
trends, labels, axes, or relationships, so it can be understood without seeing the image.
If the image contains no figure or diagram, reply with "None".
"""

    if path is None:
        return ""

    path.seek(0)
    image = path.read()

    description = ollama.chat(
        model="qwen2.5vl:latest",
        messages=[{"role": "user", "content": prompt, "images": [image]}],
    )['message']['content']

    return description

@inst.timed("fusion of extracted text")
def fuse(transcript, slides, notes, figure):
    parts = []
    components = [
        {'name': "TRANSCRIPT", 'content': transcript}, 
        {'name': "SLIDE", 'content': slides}, 
        {'name': "NOTES", 'content': notes}, 
        {'name': "FIGURE", 'content': figure}
    ]

    for component in components:
        content = component['content'].strip()
        if content and content.lower() != "none":
            parts.append(f"[{component['name']}]\n{content}")

    fusion = '\n\n'.join(parts)

    return fusion

@inst.timed("summarisation")
def make_summary(combined):
    prompt = f"""Summarise the following text  into a summary.
Use markdown headings to organise the summary into sections.
Use ONLY the provided text. Mention ALL of the topics in the text.

    TEXT:
    {combined}
    """

    summary = ollama.chat(
        model="llama3.2", 
        messages=[{"role": "user", 
                   "content": prompt}],
    )['message']['content']

    return summary

def generate_ques(content, spec, retries=3):
    prompt = f"""You are a quiz generator. Read the content, then create ONE {spec["type"]} question.

CONTENT:
{content}

Now generate ONE {spec["type"]} question about the content above.
{spec["instructions"]}
Return ONLY a JSON object with these EXACT keys: "question", "answer", "options".
Example shape: {json.dumps(spec["shape"])}
Do NOT return a summary. Do NOT use any other keys."""
    
    ques_type = spec["type"]
    q = {}

    for retry in range(retries):
        response = ollama.chat(model="llama3.2", 
                               messages=[
                                   {"role": "system", "content": "You generate quiz questions as JSON. You never summarise."},
                                   {"role": "user", "content": prompt}
                                ],
                                format="json",
                                )
        
        try:
            q = json.loads(response["message"]["content"])
        except json.JSONDecodeError:
            print(f"Failed at attempt {retry}")
            continue

        ques = q.get("question")
        ans = q.get("answer")

        print(f"Question: {ques}")
        print(f"Answer: {ans}")

        if ques and ans not in (None, "", []):
            if (ques_type == "fill-in-the-blank" and "_____" in ques and (ans.lower() not in ques.lower())) or ques_type != "fill-in-the-blank":
                return {
                    "type": ques_type, 
                    "question": ques,
                    "answer": ans,
                    "options": q.get("options")
                    }
        
    return {
            "type": ques_type,
            "question": q.get("question"),
            "answer": None,
            "options": None,
            "failed": True
            }

@inst.timed("create quiz")
def make_quiz(combined):
    questions = []

    qspecs = [
        {
            "type": "multiple-choice",
            "shape": {"question": "<question>", "answer": "<correct answer>", "options": ["<option1>","<option2>","<option3>","<option4>"]},
            "instructions": "Provide 4 options with exactly one correct. 'answer' is the exact correct option."
        }, 
        {
            "type": "fill-in-the-blank",
            "shape": {"question": "<sentence with a missing word represented as _____>", "answer": "<missing word>", "options": None},
            "instructions": "Write a factual sentence, replacing ONE key term with '_____'. The 'question' is the sentence WITH the blank. The 'answer' is the exact word you removed. The answer word must NOT appear anywhere in the question."
        }, 
        {
            "type": "true-false", 
            "shape": {"question": "<question>", "answer": "<correct answer>", "options": None},
            "instructions": "Write a declarative statement. 'answer' is either 'true' or 'false'."
        }, 
        {
            "type": "short-answer",
            "shape": {"question": "<question>", "answer": "<answer>", "options": None},
            "instructions": "Write a question. 'answer' is a complete correct sentence."
        }
    ]

    for spec in qspecs:
        q = generate_ques(combined, spec)
        if not q.get("failed"):
            questions.append(q)

    return questions

@inst.timed("extract topics")
def extract_topics(text, n=8):
    prompt = (
        "You are labelling the topics covered in a lecture.\n"
        f"List at most {n} distinct topics.\n"
        "Each topic must be 1-4 words and specific to the material.\n"
        'Respond with JSON only: {"topics": ["...", "..."]} \n\n'
        f"Lecture material:\n{text[:6000]}"
    )

    for _ in range(3):
        raw = ollama.generate(
            model="llama3.2", prompt=prompt, format="json",
            options={"temperature": 0.1, "seed": 42},
        )["response"]

        try:
            topics = [str(t).strip() for t in json.loads(raw)["topics"] if str(t).strip()]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

        seen, out = set(), []
        for t in topics:
            if t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)

        if out:
            return out[:n]

    return ["General"]

@inst.timed("tag topics")
def tag_topic(question, topics):

    if not topics:
        return "General"

    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(topics, 1))
    prompt = (
        f"Which topic does this question belong to?\n{numbered}\n\n"
        f"Question: {question}\n\n"
        "Respond with JSON only, using the topic's exact text (not its number): "
        '{"topic": "<exact topic text>"}'
    )

    raw = ollama.generate(
        model="llama3.2", prompt=prompt, format="json",
        options={"temperature": 0.0, "seed": 42},
    )["response"]

    try:
        topic = str(json.loads(raw)["topic"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return "Unclassified"

    if topic.isdigit():
        idx = int(topic) - 1
        if 0 <= idx < len(topics):
            return topics[idx]
        return "Unclassified"

    return topic

@inst.timed("generate score")
def generate_score(results):
    times_per_word = [r["time"] / r["num_words"] for r in results]
    mean = statistics.mean(times_per_word)
    stdev = statistics.stdev(times_per_word) if len(times_per_word) > 1 else 0

    for r, tpw in zip(results, times_per_word):
        correct = str(r["user_answer"]).strip().lower() == str(r["correct_answer"]).strip().lower()
        r["correct"] = correct
        z = (tpw - mean) / stdev if stdev > 0 else 0
        r["confidence"] = 0.0 if not correct else 1 / (1 + math.exp(z))

    topic_data = defaultdict(lambda: {"correct": 0, "total": 0, "confidences": []})
    for r in results:
        t = topic_data[r["topic"]]
        t["total"] += 1
        t["correct"] += int(r["correct"])
        t["confidences"].append(r["confidence"])

    performance = []
    for topic, d in topic_data.items():
        performance.append({
            "topic": topic,
            "score": f"{d['correct']}/{d['total']}",
            "confidence": round(sum(d["confidences"]) / len(d["confidences"]), 2),
        })

    performance.sort(key=lambda x: x["confidence"])
    return json.dumps(performance, indent=2)

@inst.timed("generate feedback")
def generate_feedback(performance, combined): 
    prompt = f"""A student completed a quiz. Their performance per topic is provided
    (confidence is 0-1 where low means they were wrong or hesitant)
    Write encouraging, specific feedback: acknowledge strong topics and gently 
    highlight weak ones. 2-3 short paragraphs. 
    Stick strictly to the performance. Suggest improvements to weaknesses using the provided content.

    PERFORMANCE:
    {performance}

    CONTENT:
    {combined}
    """

    feedback = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", 
                   "content": prompt}],
    )["message"]["content"]

    return feedback

@inst.timed("create plan")
def create_plan(performance, combined): 
    prompt = f"""A student completed a quiz. Their performance per topic is provided.
    (confidence 0-1; low = weak):
    Create a prioritised study plan. Focus most on the lowest-confidence topics.
    For each topic that needs work, give a concrete, specific study action.
    Order from highest to lowest priority. Keep it actionable.
    Make learning suggestions based on the provided content.

    PERFORMANCE:
    {performance}

    CONTENT:
    {combined}
    """

    plan = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user",
                   "content": prompt}],
    )["message"]["content"]

    return plan

LLM_MODEL = "llama3.2"

def release_llm(model=LLM_MODEL, label="run"):
    try:
        ollama.generate(model=model, prompt="", keep_alive=0)
    except Exception as exc:
        print(f"[release_llm] could not unload {model}: {exc}")
        return False

    if model == LLM_MODEL:
        inst.report()
        inst.save_run(label)
        inst.reset()

    return True
