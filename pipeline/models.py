from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import gc
import torch

_models = {}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get(name, loader):
    """
    loading cache for Whisper, TrOCR
    """
    if name not in _models:
        _models[name] = loader()
    return _models[name]

def unload_all():
    """ 
    releases the ingestion models
    called once ingestion is finished 
    """
    _models.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_trocr():
    """
    loads TrOCR once
    """
    name = 'microsoft/trocr-base-handwritten'
    processor = TrOCRProcessor.from_pretrained(name)
    model = VisionEncoderDecoderModel.from_pretrained(name).to(DEVICE)
    model.eval()
    return processor, model