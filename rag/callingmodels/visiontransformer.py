# no the visual_crops i got will passed thorugh the clip model and will be turened to embeddings here, a lot more efficient that google gemini vioson and ocr models inc ase of diagrams , pcitures etcd


import torch
from transformers import CLIPProcessor, CLIPModel #clip model import
from preprocessing.splitting import extract_visual_crops

# Load pre-trained CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def get_clip_image_embedding(pil_image):
    """Generates a normalized vector embedding for a diagram or doodle using CLIP."""
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.squeeze(0).numpy()

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
