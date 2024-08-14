import os
import torch
from typing import Literal, Tuple

from PIL import Image
from torchvision import transforms

from modules.modelloader import load_file_from_url
try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        models_path = os.path.abspath("models")


usage_to_weights_file = {
    'General': 'BiRefNet',
    'General-Lite': 'BiRefNet_T',
    'Portrait': 'BiRefNet-portrait',
    'DIS': 'BiRefNet-DIS5K',
    'HRSOD': 'BiRefNet-HRSOD',
    'COD': 'BiRefNet-COD',
    'DIS-TR_TEs': 'BiRefNet-DIS5K-TR_TEs'
}


class ImagePreprocessor():
    def __init__(self, resolution: Tuple[int, int] = (1024, 1024)) -> None:
        self.transform_image = transforms.Compose([
            # transforms.Resize(resolution),    # 1. keep consistent with the cv2.resize used in training 2. redundant with that in path_to_image()
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image: Image.Image) -> torch.Tensor:
        image = self.transform_image(image.convert('RGB'))
        return image
    

def get_model_path(model_name: Literal['General', 'General-Lite', 'Portrait', 'DIS', 'HRSOD', 'COD', 'DIS-TR_TEs']):
    return os.path.join(models_path, "birefnet", f"{model_name}.safetensors")


def download_models(model_root, model_urls):
    if not os.path.exists(model_root):
        os.makedirs(model_root, exist_ok=True)
    
    for local_file, url in model_urls:
        local_path = os.path.join(model_root, local_file)
        if not os.path.exists(local_path):
            load_file_from_url(url, model_dir=model_root, file_name=local_file)


def download_birefnet_model(model_name: Literal['General', 'General-Lite', 'Portrait', 'DIS', 'HRSOD', 'COD', 'DIS-TR_TEs']):
    """
    Downloading birefnet model from huggingface.
    """
    model_root = os.path.join(models_path, "birefnet")
    model_urls = (
        (f"{model_name}.safetensors", f"https://huggingface.co/ZhengPeng7/{usage_to_weights_file[model_name]}/resolve/main/model.safetensors"),
    )
    download_models(model_root, model_urls)