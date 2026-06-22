from models import VisionLanguageModel,LanguageModel
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMultimodalLM
)

VLMs = {
    "llava-hf/llava-1.5-7b-hf": (
        AutoProcessor, 
        LlavaForConditionalGeneration
    ),
    "Qwen/Qwen2.5-VL-7B-Instruct": (
        AutoProcessor, 
        Qwen2_5_VLForConditionalGeneration
    ),
    "llava-hf/llama3-llava-next-8b-hf": (
        LlavaNextProcessor, 
        LlavaNextForConditionalGeneration
        
    ),
    "google/gemma-4-31B-it": (
        AutoProcessor,
        AutoModelForMultimodalLM
    ),

    "Qwen/Qwen3-VL-30B-A3B-Instruct": (
        AutoProcessor,
        Qwen3VLMoeForConditionalGeneration
    ),
    "google/gemma-4-26B-A4B-it": (
        AutoProcessor,
        AutoModelForMultimodalLM
    ),
    "Qwen/Qwen3-VL-32B-Instruct": (
        AutoProcessor,
        Qwen3VLForConditionalGeneration
    ),
}

LMs = {
    "meta-llama/Llama-3.1-8B-Instruct": (
        AutoTokenizer,
        AutoModelForCausalLM
    ),
    "Qwen/Qwen2.5-7B-Instruct": (
        AutoTokenizer,
        AutoModelForCausalLM
    ),
    "tiiuae/Falcon3-7B-Instruct": (
        AutoTokenizer,
        AutoModelForCausalLM
    ),
}

def get_model(model_name: str, device: str, logger, **kwargs):
    """
    Returns an instance of BaseModel (LanguageModel o VisionLanguageModel)
    selecting by the name. 
    Args:
        model_name (str): the hugginface name of the model
        processor or tokenizer: The processor (VLM) or the tokenizer (LM) of the model
        model_class: The class of appartenence of the model
        device: which GPU to use (or eventually the CPU)
        logger: instance of the logger
        kwargs: model parameters (top_p, top_k, temperature)
    """
    if model_name in VLMs:
        processor, model_class = VLMs[model_name]
        return VisionLanguageModel(
            model_name, 
            processor, 
            model_class, 
            device, logger, **kwargs
        )
    if model_name in LMs:
        tokenizer, model_class = LMs[model_name]
        return LanguageModel(
            model_name, 
            tokenizer, 
            model_class, 
            device=device, logger=logger,**kwargs
        )
    raise ValueError(f"Unknown model: {model_name}")
