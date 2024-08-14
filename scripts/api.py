import os
import torch
import base64
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any, Literal
import gradio as gr
from PIL import Image
import numpy as np
import safetensors.torch

from birefnet.models.birefnet import BiRefNet
from internal_birefnet.utils import ImagePreprocessor, download_birefnet_model, get_model_path
from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from modules.devices import torch_gc


def decode_to_pil(image):
    if os.path.exists(image):
        return Image.open(image)
    elif type(image) is str:
        return decode_base64_to_image(image)
    elif type(image) is Image.Image:
        return image
    elif type(image) is np.ndarray:
        return Image.fromarray(image)
    else:
        Exception("Not an image")


def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return encode_pil_to_base64(image).decode()
    elif type(image) is np.ndarray:
        pil = Image.fromarray(image)
        return encode_pil_to_base64(pil).decode()
    else:
        Exception("Invalid type")


birefnet: BiRefNet | None = None
birefnet_model_name: str | None = None

def clear_model_cache():
    global birefnet
    birefnet = None
    torch_gc()


def get_device(device_id: int, flag_force_cpu: bool):
    if flag_force_cpu:
        device = "cpu"
    else:
        try:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda:' + str(device_id)
            else:
                device = "cpu"
        except:
            device = "cpu"
    return device


def init_birefnet(model_name: Literal['General', 'General-Lite', 'Portrait', 'DIS', 'HRSOD', 'COD', 'DIS-TR_TEs'], device: str):
    weight_path = get_model_path(model_name)
    
    state_dict = safetensors.torch.load_file(weight_path, device=device)

    model = BiRefNet(bb_pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def get_model_using_cache(
        model_name: Literal['General', 'General-Lite', 'Portrait', 'DIS', 'HRSOD', 'COD', 'DIS-TR_TEs'],
        device: str,
        use_model_cache: bool):

    global birefnet
    if not use_model_cache:
        clear_model_cache()
        return init_birefnet(model_name, device)
    if not birefnet or not birefnet_model_name or birefnet_model_name != model_name:
        clear_model_cache()
        birefnet = init_birefnet(model_name, device)
    return birefnet


def is_file(input):
    return os.path.exists(input) or (type(input) is str and (input.startswith("http://") or input.startswith("https://")))


def get_output_path(output_dir):
    if os.path.isabs(output_dir):
        return output_dir
    from modules.paths_internal import data_path
    return os.path.join(data_path, output_dir)


def birefnet_api(_: gr.Blocks, app: FastAPI):

    class BiRefNetRequest(BaseModel):
        image: str = ""
        resolution: str = ""
        model_name: Literal['General', 'General-Lite', 'Portrait', 'DIS', 'HRSOD', 'COD', 'DIS-TR_TEs']
        output_dir: str = 'outputs/birefnet/'  # directory to save output image
        device_id: int = 0  # gpu device id
        send_output: bool = True
        save_output: bool = False
        use_model_cache: bool = True
        flag_force_cpu: bool = False

    def fast_check_birefnet_args(payload: BiRefNetRequest):
        if not payload.image:
            raise ValueError("Input image is not optional")

    @app.post("/birefnet/single")
    async def execute_birefnet_single(payload: BiRefNetRequest = Body(...)) -> Any:
        print("BiRefNet API /birefnet/single received request")

        fast_check_birefnet_args(payload)

        download_birefnet_model(payload.model_name)

        image = decode_to_pil(payload.image).convert("RGBA")

        resolution = f"{image.width}x{image.height}" if payload.resolution == '' else payload.resolution
        resolution = [int(int(reso)//32*32) for reso in resolution.strip().split('x')]

        image_shape = image.size[::-1]
        image_pil = image.resize(tuple(resolution))

        image_preprocessor = ImagePreprocessor(resolution=tuple(resolution))
        image_proc = image_preprocessor.proc(image_pil)
        image_proc = image_proc.unsqueeze(0)

        device = get_device(payload.device_id, payload.flag_force_cpu)

        birefnet = get_model_using_cache(
            payload.model_name,
            device,
            payload.use_model_cache
        )

        with torch.no_grad():
            scaled_pred_tensor = birefnet(image_proc.to(device))[-1].sigmoid()

        if device.startswith('cuda'):
            scaled_pred_tensor = scaled_pred_tensor.cpu()

        pred = torch.nn.functional.interpolate(scaled_pred_tensor, size=image_shape, mode='bilinear', align_corners=True).squeeze().numpy()

        pred_rgba = np.zeros((*pred.shape, 4), dtype=np.uint8)
        pred_rgba[..., :3] = (pred[..., np.newaxis] * 255).astype(np.uint8)
        pred_rgba[..., 3] = (pred * 255).astype(np.uint8)

        image_array = np.array(image)
        image_pred = image_array * (pred_rgba / 255.0)
        
        output_image = Image.fromarray(image_pred.astype(np.uint8), 'RGBA')

        if payload.save_output:
            output_folder = get_output_path(payload.output_dir)
            os.makedirs(output_folder, exist_ok=True)
            input_file_name = os.path.splitext(os.path.basename(payload.image))[0] if is_file(payload.image) else "output"
            output_path = os.path.join(output_folder, f"{input_file_name}_no_background.png")
            output_image.save(output_path)

        if payload.send_output:
            output_image_base64 = encode_to_base64(output_image)
        else:
            output_image_base64 = None
        
        print("BiRefNet API /birefnet/single finished")

        return {"output_image": output_image_base64}

        
try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(birefnet_api)
except:
    print("BiRefNet API failed to initialize")