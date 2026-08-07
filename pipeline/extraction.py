import ollama
from faster_whisper import WhisperModel
from PIL import Image
import cv2
import numpy as np
import fitz
from pptx import Presentation
import io
import instrumentation as inst
from .models import get, load_trocr
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@inst.timed("transcription")
def transcribe_audio(path):
    if path is None:
        return ""

    whisper_model = get("whisper", lambda: WhisperModel("base", compute_type="int8"))

    segments, info = whisper_model.transcribe(path, beam_size=1, vad_filter=True)
    transcript = " ".join(segment.text for segment in segments)

    return transcript

@inst.timed("slide extraction")
def extract_slides(path):
    if path is None:
        return ""
    
    name = path.name.lower()

    # PDF
    if name.endswith('.pdf'):
        data = path.read()
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    # PPTX
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

    # other / incorrect file types do not get parsed here
    else:
        return ""

@inst.timed("OCR")
def ocr_notes(path):
    if path is None:
        return ""

    trocr_processor, trocr_model = get("trocr", load_trocr)

    # minimum size of one box
    min_area = 0.0004
    pad = 6

    image = Image.open(path).convert("RGB")
    img = np.array(image.convert("L")) # convert to grayscale
    h, w = img.shape # get height and width
    mean_brightness = img.mean() # get brightness

    # if background is dark with light text then do not invert
    if mean_brightness < 127:         
        thresh_type = cv2.THRESH_BINARY  

    # if background is light with dark text then invert     
    else:                            
        thresh_type = cv2.THRESH_BINARY_INV   

    _, binary = cv2.threshold(img, 0, 255, thresh_type + cv2.THRESH_OTSU)

    # dilate white pixels horizontally so letters in a single line merge
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 20), 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # get shape outlines for the merged white pixels
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours] # create bounding boxes around the shapes
    boxes = [b for b in boxes if b[2] * b[3] > min_area * w * h] # only keep boxes larger than the minimum area
    boxes.sort(key=lambda b: b[1]) # sort boxes from top to bottom

    # crop each box into a separate image
    lines = []
    for x, y, bw, bh in boxes:
        lines.append(image.crop((max(0, x - pad), max(0, y - pad),
                                    min(w, x + bw + pad), min(h, y + bh + pad))))

    # apply TrOCR to all lines in a single batch instead of one generate() call per line
    notes = ""
    if lines:
        pixel_values = trocr_processor(
            images=[line.convert("RGB") for line in lines], return_tensors="pt"
        ).pixel_values.to(DEVICE)
        generated_ids = trocr_model.generate(pixel_values)
        decoded = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)
        notes = "\n".join(decoded) + "\n"

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
    image = Image.open(path).convert("RGB")

    # downscale large images
    max_side = 1024
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="PNG")

    # generate description
    description = ollama.chat(
        model="qwen2.5vl:latest",
        messages=[{"role": "user", "content": prompt, "images": [buf.getvalue()]}],
    )['message']['content']

    return description