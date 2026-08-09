import subprocess
import tempfile
from pathlib import Path
import cv2
import pdfplumber
from pptx import Presentation
from PIL import Image
import os
import time

AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.aac', '.flac', '.wma', '.ogg')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv')
DOCUMENT_EXTENSIONS = ('.pdf', '.pptx', '.ppt')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

# --> AUDIO EXTRACTION
def extract_from_audio(audio_path):
    return str(audio_path)

# --> VIDEO EXTRACTION

def extract_from_video(video_path):
    """
    extract audio and slide frames from video
    """
    video_path = str(video_path)
    
    # extract audio
    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    result = subprocess.run([
        "ffmpeg", "-i", video_path, "-q:a", "9",
        "-acodec", "pcm_s16le", audio_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError(f"Audio extraction produced empty file: {audio_path}")
    
    # extract slide frames (scene detection)
    frames = []
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # detect scene change
        if prev_frame is not None:
            diff = cv2.absdiff(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            ).mean()
            
            if diff > 10:  # threshold for scene change
                frames.append(frame)
        
        prev_frame = frame
        frame_count += 1
    
    cap.release()
    
    return audio_path, frames


# --> PDF EXTRACTION

def extract_from_pdf(pdf_path):
    """
    extract text and images/figures from PDF
    """
    pdf_path = str(pdf_path)
    text_parts = []
    images = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # extract text
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            
            # extract images
            for img in page.images:
                images.append(img)
    
    text = "\n\n".join(text_parts)
    return text, images


# --> PPTX EXTRACTION

def extract_from_pptx(pptx_path):
    """
    extract text and images/figures from pptx
    """
    pptx_path = str(pptx_path)
    text_parts = []
    images = []
    
    prs = Presentation(pptx_path)
    
    for slide_idx, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            # extract text
            if hasattr(shape, "text") and shape.text.strip():
                text_parts.append(shape.text)
            
            # extract images
            if shape.shape_type == 13:  # image shape
                try:
                    image = shape.image
                    images.append(image)
                except Exception:
                    pass
    
    text = "\n\n".join(text_parts)
    return text, images


# --> UNIFIED ROUTER

def extract_from_file(file_obj):
    """
    route file to correct extraction function based on type
    returns:
        dict with keys: audio, text, images, slide_frames
    """
    filename = file_obj.name.lower()
    
    result = {
        "audio": None,
        "text": "",
        "images": [],
        "slide_frames": []
    }
    
    # save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
        tmp.write(file_obj.getbuffer() if hasattr(file_obj, 'getbuffer') else file_obj.read())
        tmp_path = tmp.name
    
    try:
        if any(filename.endswith(ext) for ext in VIDEO_EXTENSIONS):
            # video
            audio, frames = extract_from_video(tmp_path)
            result['audio'] = audio
            result['slide_frames'] = frames

        elif any(filename.endswith(ext) for ext in AUDIO_EXTENSIONS):
            # audio
            result['audio'] = extract_from_audio(tmp_path)
        
        elif filename.endswith('.pdf'):
            # PDF
            text, images = extract_from_pdf(tmp_path)
            result['text'] = text
            result['images'] = images
        
        elif filename.endswith(('.pptx', '.ppt')):
            # pptx
            text, images = extract_from_pptx(tmp_path)
            result['text'] = text
            result['images'] = images
        
        elif any(filename.endswith(ext) for ext in IMAGE_EXTENSIONS):
            # single image (handwritten notes)
            image = Image.open(tmp_path)
            result['images'] = [image]
        
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        
    finally:
        # give everything time to release the file
        time.sleep(0.1)
        
        # Now delete
        try:
            # clean up temp file
            # unless it is an audio file because that needs to be kept for transcription
            if os.path.exists(tmp_path) and not any(filename.endswith(ext) for ext in AUDIO_EXTENSIONS + VIDEO_EXTENSIONS):
                os.remove(tmp_path)
        except PermissionError:
            # if it still fails, let system cleanup on reboot
            pass
    
    return result