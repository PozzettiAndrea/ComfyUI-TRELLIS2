from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image

# Remap gated models to public alternatives
RMBG_MODEL_REMAP = {
    "briaai/RMBG-2.0": "ZhengPeng7/BiRefNet",
}


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # Remap gated models to public reuploads
        actual_model_name = RMBG_MODEL_REMAP.get(model_name, model_name)
        if actual_model_name != model_name:
            print(f"[ComfyUI-TRELLIS2] Remapping {model_name} -> {actual_model_name}")
        print(f"[ComfyUI-TRELLIS2] Loading BiRefNet model: {actual_model_name}...")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            actual_model_name, trust_remote_code=True
        )
        print(f"[ComfyUI-TRELLIS2] BiRefNet model loaded successfully")
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    