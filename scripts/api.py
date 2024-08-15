import datetime
import os
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any
import gradio as gr
from PIL import Image
import numpy as np

from internal_birefnet.pipeline import BiRefNetModelName, BiRefNetPipeline
from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from modules.devices import torch_gc


birefnet: BiRefNetPipeline | None = None


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


def clear_model_cache():
    global birefnet
    birefnet = None
    torch_gc()


def get_pipeline_using_cache(model_name: BiRefNetModelName, device_id: int, flag_force_cpu: bool, use_model_cache: bool):
    global birefnet
    if not use_model_cache:
        clear_model_cache()
        return BiRefNetPipeline(model_name, device_id, flag_force_cpu)
    if not birefnet or birefnet.model_name != model_name or birefnet.device_id != device_id or birefnet.flag_force_cpu != flag_force_cpu:
        clear_model_cache()
        birefnet = BiRefNetPipeline(model_name, device_id, flag_force_cpu)
    return birefnet


def is_file(input):
    return os.path.exists(input) or (type(input) is str and (input.startswith("http://") or input.startswith("https://")))


def get_output_path(output_dir):
    today = f"{datetime.date.today()}"
    if os.path.isabs(output_dir):
        return os.path.join(output_dir, today)
    from modules.paths_internal import data_path
    return os.path.join(data_path, output_dir, today)


def birefnet_api(_: gr.Blocks, app: FastAPI):
    class BiRefNetRequest(BaseModel):
        image: str = ""
        resolution: str = ""
        model_name: BiRefNetModelName
        output_dir: str = 'outputs/birefnet/'  # directory to save output image
        device_id: int = 0  # gpu device id
        send_output: bool = True
        save_output: bool = False
        use_model_cache: bool = True
        flag_force_cpu: bool = False


    @app.post("/birefnet/single")
    async def execute_birefnet_single(payload: BiRefNetRequest = Body(...)) -> Any:
        print("BiRefNet API /birefnet/single received request")

        image = decode_to_pil(payload.image)
        if image is None:
            raise ValueError("Input image is not optional")

        birefnet = get_pipeline_using_cache(
            payload.model_name,
            payload.device_id,
            payload.flag_force_cpu,
            payload.use_model_cache
        )
        
        output_image = birefnet.process(image, payload.resolution)

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