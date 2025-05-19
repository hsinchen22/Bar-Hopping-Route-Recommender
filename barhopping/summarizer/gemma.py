import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from barhopping.config import GEMMA_MODEL, HF_TOKEN
from barhopping.logger import logger

def get_device():
    if torch.backends.mps.is_available():
        logger.info("Using MPS device")
        return torch.device("mps")
    else:
        logger.info("Using CPU device")
        return torch.device("cpu")

processor = AutoProcessor.from_pretrained(
    GEMMA_MODEL, use_auth_token=HF_TOKEN
)
model = Gemma3ForConditionalGeneration.from_pretrained(
    GEMMA_MODEL,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float32
).eval()

model = model.to(get_device())

def summarize_bar(reviews: list[str], photos: list[str]) -> str:
    try:
        device = get_device()
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it").to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        
        # Prepare input
        reviews_text = "\n".join(reviews)
        photos_text = "\n".join(photos) if photos else "No photos available"
        
        prompt = f"""Based on the following reviews and photos, provide a concise summary of this bar:

Reviews:
{reviews_text}

Photos:
{photos_text}

Summary:"""
        
        # Generate summary
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.split("Summary:")[-1].strip()
        
    except Exception as e:
        logger.error(f"Error in summarize_bar: {e}")
        if "out of memory" in str(e).lower():
            logger.warning("Switching to CPU due to memory issues")
            torch.cuda.empty_cache()
            return summarize_bar(reviews, photos)
        return "Unable to generate summary at this time."
