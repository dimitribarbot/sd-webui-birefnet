import datetime
from typing import Any, Optional
import os

from fastapi import FastAPI, Body
from pydantic import BaseModel
import gradio as gr
from PIL import Image
import numpy as np

from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from modules.devices import torch_gc

from internal_birefnet.pipeline import BiRefNetModelName, BiRefNetPipeline


birefnet: Optional[BiRefNetPipeline] = None


def decode_to_pil(image):
    if os.path.exists(image):
        return Image.open(image)
    if isinstance(image, str):
        return decode_base64_to_image(image)
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise Exception("Not an image")


def encode_to_base64(image):
    if isinstance(image, str):
        return image
    if isinstance(image, Image.Image):
        return encode_pil_to_base64(image).decode()
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
        return encode_pil_to_base64(pil).decode()
    raise Exception("Invalid type")


def clear_model_cache():
    global birefnet
    birefnet = None
    torch_gc()


def get_pipeline_using_cache(
    model_name: BiRefNetModelName,
    device_id: int,
    flag_force_cpu: bool,
    use_model_cache: bool,
    use_fp16: bool,
):
    global birefnet
    if not use_model_cache:
        clear_model_cache()
        return BiRefNetPipeline(model_name, device_id, flag_force_cpu)
    if (
        not birefnet
        or birefnet.model_name != model_name
        or birefnet.device_id != device_id
        or birefnet.flag_force_cpu != flag_force_cpu
        or birefnet.use_fp16 != use_fp16
    ):
        clear_model_cache()
        birefnet = BiRefNetPipeline(model_name, device_id, flag_force_cpu, use_fp16)
    return birefnet


def is_file(input_file):
    return os.path.exists(input_file) or (
        isinstance(input_file, str)
        and (input_file.startswith("http://") or input_file.startswith("https://"))
    )


def get_output_path(output_dir):
    today = f"{datetime.date.today()}"
    if os.path.isabs(output_dir):
        return os.path.join(output_dir, today)
    from modules.paths_internal import data_path

    return os.path.join(data_path, output_dir, today)


def save_image_file(
    image: Image.Image, folder: str, base_filename: str, extension: str
):
    if extension.lower() in ("jpg", "jpeg", "webp"):
        image = image.convert("RGB")

    output_path = os.path.join(folder, f"{base_filename}.{extension}")

    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(folder, f"{base_filename}_{counter}.{extension}")
        counter += 1
    image.save(output_path)


def process_image(
    pipeline: BiRefNetPipeline,
    image_input: str,
    resolution: str,
    return_foreground: bool,
    return_mask: bool,
    return_edge_mask: bool,
    edge_mask_width: int,
    output_dir: str,
    save_output: bool,
    send_output: bool,
    default_output_filename: str,
    output_extension: str,
):
    image = decode_to_pil(image_input)
    if image is None:
        raise ValueError("Input image is not optional")

    mask, foreground, edge_mask = pipeline.process(
        image.convert("RGB"),
        resolution,
        return_mask,
        return_foreground,
        return_edge_mask,
        edge_mask_width,
    )

    if save_output:
        output_folder = get_output_path(output_dir)
        os.makedirs(output_folder, exist_ok=True)
        input_file_name = (
            os.path.splitext(os.path.basename(image_input))[0]
            if is_file(image_input)
            else default_output_filename
        )
        if foreground:
            save_image_file(
                foreground,
                output_folder,
                f"{input_file_name}-foreground",
                output_extension,
            )
        if mask:
            save_image_file(
                mask,
                output_folder,
                f"{input_file_name}-foreground-mask",
                output_extension,
            )
        if edge_mask:
            save_image_file(
                edge_mask,
                output_folder,
                f"{input_file_name}-foreground-edge-mask",
                output_extension,
            )

    if send_output:
        mask_base64 = encode_to_base64(mask) if mask else None
        output_image_base64 = encode_to_base64(foreground) if foreground else None
        edge_mask_base64 = encode_to_base64(edge_mask) if edge_mask else None
    else:
        mask_base64 = None
        output_image_base64 = None
        edge_mask_base64 = None

    return mask_base64, output_image_base64, edge_mask_base64


def birefnet_api(_: gr.Blocks, app: FastAPI):
    class BiRefNetSingleRequest(BaseModel):
        image: str = ""
        resolution: str = ""
        model_name: BiRefNetModelName
        return_foreground: bool = True
        return_mask: bool = True
        return_edge_mask: bool = True
        edge_mask_width: int = 64
        output_dir: str = "outputs/birefnet/"  # directory to save output image
        output_extension: str = "png"
        device_id: int = 0  # gpu device id
        send_output: bool = True
        save_output: bool = False
        use_model_cache: bool = True
        flag_force_cpu: bool = False
        use_fp16: bool = True

    @app.post("/birefnet/single")
    async def execute_birefnet_single(
        payload: BiRefNetSingleRequest = Body(...),
    ) -> Any:
        print("BiRefNet API /birefnet/single received request")

        pipeline = get_pipeline_using_cache(
            payload.model_name,
            payload.device_id,
            payload.flag_force_cpu,
            payload.use_model_cache,
            payload.use_fp16,
        )

        mask_base64, output_image_base64, edge_mask_base64 = process_image(
            pipeline,
            payload.image,
            payload.resolution,
            payload.return_foreground,
            payload.return_mask,
            payload.return_edge_mask,
            payload.edge_mask_width,
            payload.output_dir,
            payload.save_output,
            payload.send_output,
            "output",
            payload.output_extension,
        )

        print("BiRefNet API /birefnet/single finished")

        return {
            "mask": mask_base64,
            "output_image": output_image_base64,
            "edge_mask": edge_mask_base64,
        }

    class BiRefNetInput(BaseModel):
        image: str = ""
        resolution: str = ""

    class BiRefNetMultiRequest(BaseModel):
        inputs: list[BiRefNetInput]
        model_name: BiRefNetModelName
        return_foreground: bool = True
        return_mask: bool = True
        return_edge_mask: bool = True
        edge_mask_width: int = 64
        output_dir: str = "outputs/birefnet/"  # directory to save output image
        output_extension: str = "png"
        device_id: int = 0  # gpu device id
        send_output: bool = True
        save_output: bool = False
        use_model_cache: bool = True
        flag_force_cpu: bool = False
        use_fp16: bool = True

    @app.post("/birefnet/multi")
    async def execute_birefnet_multi(payload: BiRefNetMultiRequest = Body(...)) -> Any:
        print("BiRefNet API /birefnet/multi received request")

        if payload.inputs is None or len(payload.inputs) == 0:
            raise ValueError("Input images are not optional")

        pipeline = get_pipeline_using_cache(
            payload.model_name,
            payload.device_id,
            payload.flag_force_cpu,
            payload.use_model_cache,
            payload.use_fp16
        )

        count = 1
        outputs = []
        for payload_input in payload.inputs:
            try:
                mask_base64, output_image_base64, edge_mask_base64 = process_image(
                    pipeline,
                    payload_input.image,
                    payload_input.resolution,
                    payload.return_foreground,
                    payload.return_mask,
                    payload.return_edge_mask,
                    payload.edge_mask_width,
                    payload.output_dir,
                    payload.save_output,
                    payload.send_output,
                    f"output_{count}",
                    payload.output_extension,
                )

                if mask_base64:
                    outputs.append(
                        {
                            "mask": mask_base64,
                            "output_image": output_image_base64,
                            "edge_mask": edge_mask_base64,
                        }
                    )

                count += 1
            except Exception as e:
                print(f"Error processing image {count}: {str(e)}")
                continue

        print("BiRefNet API /birefnet/multi finished")

        return {"outputs": outputs}


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(birefnet_api)
except:
    print("BiRefNet API failed to initialize")
