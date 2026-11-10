import torch
from transformers import CLIPProcessor, CLIPModel
from preprocessing.splitting import extract_visual_crops

# Global cache so we don't reload multiple times
_clip_model = None
_clip_processor = None

def get_clip_models():
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        model_name = "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(model_name, low_cpu_mem_usage=True)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
    return _clip_model, _clip_processor

def get_clip_image_embedding(pil_image):
    """Generates a normalized 512-d vector embedding for a diagram using CLIP."""
    model, processor = get_clip_models()
    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        output = model.get_image_features(**inputs)
        image_features = output.pooler_output if hasattr(output, "pooler_output") else output

    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.squeeze(0).cpu().numpy().tolist()

def process_visual_path(page_image_path):
    crops = extract_visual_crops(page_image_path)
    embeddings_data = []
    
    for item in crops:
        vector = get_clip_image_embedding(item["image"])
        embeddings_data.append({
            "box": item["box"],
            "embedding": vector,
            "crop_image": item["image"]
        })
    return embeddings_data

def get_clip_text_embedding(text_query: str) -> list[float]:
    """Embeds a user query into the 512-d CLIP space."""
    model, processor = get_clip_models()
    inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model.get_text_features(**inputs)
        text_features = output.pooler_output if hasattr(output, "pooler_output") else output

    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.squeeze(0).cpu().numpy().tolist()
