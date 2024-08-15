from math import e
import os
import torch
import time
from typing import cast, Literal
from PIL import Image
from torchvision import transforms
import numpy as np
import safetensors.torch

from birefnet.models.birefnet import BiRefNet

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

BiRefNetModelName = Literal['General', 'General-Lite', 'Portrait', 'DIS', 'HRSOD', 'COD', 'DIS-TR_TEs']

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


class ImagePreprocessor():
    def __init__(self) -> None:
        self.transform_image: transforms.Compose = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image: Image.Image) -> torch.Tensor:
        image_tf = cast(torch.Tensor, self.transform_image(image))
        return image_tf
    

class BiRefNetPipeline(object):
    def __init__(self, model_name: BiRefNetModelName='General', device_id: int=0, flag_force_cpu: bool=False):
        self.model_name = model_name
        self.device_id = device_id
        self.flag_force_cpu = flag_force_cpu

        if self.flag_force_cpu:
            self.device = "cpu"
        else:
            try:
                if torch.backends.mps.is_available():
                    self.device = 'mps'
                elif torch.cuda.is_available():
                    self.device = 'cuda:' + str(self.device_id)
                else:
                    self.device = "cpu"
            except:
                self.device = "cpu"

        download_birefnet_model(self.model_name)

        weight_path = get_model_path(self.model_name)
        
        state_dict = safetensors.torch.load_file(weight_path, device=self.device)

        self.birefnet = BiRefNet(bb_pretrained=False)
        self.birefnet.load_state_dict(state_dict)
        self.birefnet.to(self.device)
        self.birefnet.eval()

    
    def process(self, image: Image.Image, resolution=''):
        resolution = f"{image.width}x{image.height}" if resolution == '' else resolution
        resolution = [int(int(reso)//32*32) for reso in resolution.strip().split('x')]
        resolution = cast(tuple[int, int], tuple(resolution))

        image_shape = image.size[::-1]
        image_pil = image.resize(resolution)

        image_preprocessor = ImagePreprocessor()
        image_proc = image_preprocessor.proc(image_pil)
        image_proc = image_proc.unsqueeze(0)

        with torch.no_grad():
            scaled_pred_tensor = self.birefnet(image_proc.to(self.device))[-1].sigmoid()

        pred = torch.nn.functional.interpolate(scaled_pred_tensor, size=image_shape, mode='bilinear', align_corners=True).squeeze()
        pred = pred.cpu().numpy()

        pred_rgba = np.zeros((*pred.shape, 4), dtype=np.uint8)
        pred_rgba[..., :3] = (pred[..., np.newaxis] * 255).astype(np.uint8)
        pred_rgba[..., 3] = (pred * 255).astype(np.uint8)

        image_array = np.array(image.convert("RGBA"))
        image_pred = image_array * (pred_rgba / 255.0)
        
        output_image = Image.fromarray(image_pred.astype(np.uint8), 'RGBA')

        return output_image