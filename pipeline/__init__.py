from .extraction import transcribe_audio, extract_slides, ocr_notes, describe_figure
from .generation import release_llm
from .generation_feedback import generate_score, generate_feedback
from .generation_quiz import generate_ques, make_quiz, extract_topics
from .generation_summary import make_summary
from .generation_plan import create_plan, render_study_plan, _allocate_days, _priority_bucket
from .models import unload_all, _models
from .utils import fuse
from .config import TEXT_MODEL, OCR_MODEL, VL_MODEL
import ollama
import instrumentation as inst

__all__ = [
    "transcribe_audio", "extract_slides", "ocr_notes", "describe_figure", # extraction
    "release_llm", # general generation
    "create_plan", "render_study_plan", "_allocate_days", "_priority_bucket", # learning plan
    "generate_score", "generate_feedback", # feedback
    "generate_ques", "make_quiz", "extract_topics", # quiz 
    "make_summary", # summary
    "unload_all", "_models", # model-related
    "fuse", "ollama", # utility
    "inst", # instrumentation (for testing)
    "TEXT_MODEL", "OCR_MODEL", "VL_MODEL" # model constants
]