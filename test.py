from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch, torch.nn.functional as F

name = "Wvolf/ViT_Deepfake_Detection"
proc = AutoImageProcessor.from_pretrained(name)
model = AutoModelForImageClassification.from_pretrained(name).eval()

print("id2label:", model.config.id2label)

img = Image.open("test_image1.jpg").convert("RGB")
with torch.no_grad():
    logits = model(**proc(images=img, return_tensors="pt")).logits
    probs = F.softmax(logits, dim=-1)[0]
print("probs:", {model.config.id2label[i]: float(p) for i, p in enumerate(probs)})