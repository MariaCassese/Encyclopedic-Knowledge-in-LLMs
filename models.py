from abc import ABC, abstractmethod
import torch
from utils.utils import preprocess_img
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = 400_000_000 

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseModel(ABC):
    @property
    def supports_image(self) -> bool:
        # default: models don't support images
        return False
    
    @abstractmethod
    def predict(self, text: str, image_path: str) -> str:
        """Generates the answer to a query"""
        pass

class VisionLanguageModel(BaseModel):
    
    def __init__(
        self,
        model_name: str,
        processor,           
        model_class,         
        device: str,
        logger,
        *,
        do_sample: bool = True,
        top_p: float = 0.9,
        temperature: float = 0.7,
        max_new_tokens: int = 100,
    ):
        self.logger = logger
        
        # Determine device_map based on device parameter
        if device == "auto":
            device_map = "auto" 
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Loading model with device_map='auto' across {torch.cuda.device_count()} GPUs")
        else:
            device_map = None 
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Loading model on device: {self.device}")
        
        self.processor = processor.from_pretrained(model_name, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28, use_fast=True)
        
        if device_map == "auto":
            self.model = model_class.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            ).eval()
        else:
            self.model = (
                model_class.from_pretrained(model_name)
                .to(self.device)
                .eval()
            )
          
        self.kwargs = {
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }

    @property
    def supports_image(self) -> bool:
        return True  # this model supports the images

    # prompt normalization in a standard conversation
    def _ensure_conversation(self, prompt):
        """
        input can be:
          - a single dict {role, content}
          - a list of messages [{role, content}, ...]
          - a string
        It returns a list of messages
        """
        if isinstance(prompt, dict):
            conv = [prompt]
        elif isinstance(prompt, list):
            conv = prompt
        else:
            conv = [{"role": "user", "content": str(prompt)}]

        # verify that for each message, content is a list of components 
        for m in conv:
            c = m.get("content", "")
            if isinstance(c, str):
                m["content"] = [{"type": "text", "text": c}]
            elif isinstance(c, list):
                pass
            else:
                m["content"] = [{"type": "text", "text": str(c)}]
        return conv

    def predict(self, prompt: dict | list | str, image_path: str | None) -> str:
        self.logger.info(f"Start generation for image: {image_path}")  # log to debug

        conversation = self._ensure_conversation(prompt)  # prompt normalization

        # check whether there is an image
        wants_image = any(
            any(isinstance(p, dict) and p.get("type") == "image" for p in m.get("content", []))
            for m in conversation
        )

        has_img_file = isinstance(image_path, str) and image_path.strip() != ""

        prompt_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        if not isinstance(prompt_text, str):  # safety check
            raise TypeError(f"apply_chat_template must return str, got {type(prompt_text)}")

        # prepare the input tensors with or without an image
        if wants_image:
            # the prompt requires an image
            if not has_img_file: #image_path is missing
                self.logger.warning("Prompt includes an image but image_path is missing — skip the record")
                return None

            try:
                image = preprocess_img(image_path)
                if image is None:
                    self.logger.warning(f"Unsupported image format (skipped): {image_path}")
                    return None
                
                inputs = self.processor(
                    images=image,
                    text=prompt_text,
                    return_tensors="pt",
                    padding=True
                )
            except FileNotFoundError:
                self.logger.warning(f"Image not found: {image_path} — skip the record")
                return None

        else:
            # the prompt doesn't require an image
            inputs = self.processor(
                text=prompt_text,
                return_tensors="pt",
                padding=True
            )

        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        with torch.inference_mode():
            ids = self.model.generate(**inputs, **self.kwargs)

        # decode: batch_decode or tokenizer
        if hasattr(self.processor, "batch_decode"):
            response = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        elif hasattr(self.processor, "tokenizer"):
            response = self.processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        else:
            response = ids[0].tolist().__repr__()

        return response


class LanguageModel(BaseModel):
    def __init__(
        self, 
        model_name: str,
        tokenizer, 
        model_class, 
        device: str,
        logger,
        *,
        do_sample: bool = True,
        top_p: float = 0.9,
        temperature: float = 0.7,
        max_new_tokens: int = 100,
    ):
        self.logger = logger
        
        # Determine device_map based on device parameter
        if device == "auto":
            device_map = "auto"  # Distribute across all available GPUs
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Loading model with device_map='auto' across {torch.cuda.device_count()} GPUs")
        else:
            device_map = None  # Use single device
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Loading model on device: {self.device}")
        
        self.tokenizer = tokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if device_map == "auto":
            self.model = model_class.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            ).eval()
        else:
            self.model = model_class.from_pretrained(model_name).to(self.device).eval()
        
        self.kwargs = {
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }

    def predict(self, prompt: str | dict | list, image_path: str | None = None) -> str:
        self.logger.info("Start LanguageModel generation")

        if isinstance(prompt, list):
            conversation = prompt
        elif isinstance(prompt, dict):
            conversation = [prompt]
        else:
            conversation = [{"role": "user", "content": str(prompt)}]

        inputs = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **self.kwargs
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]

        output = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return output.strip()
