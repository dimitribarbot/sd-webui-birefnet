from modules import scripts_postprocessing
import gradio as gr

from modules.ui_components import FormRow, InputAccordion
from modules.devices import torch_gc

from internal_birefnet.pipeline import BiRefNetPipeline, BiRefNetModelName

models = [
    "None",
    "General",
    "General-Lite",
    "Portrait",
    "DIS",
    "HRSOD",
    "COD",
    "DIS-TR_TEs"
]

birefnet: BiRefNetPipeline | None = None


def get_pipeline_using_cache(model_name: BiRefNetModelName):
    global birefnet
    if not birefnet or birefnet.model_name != model_name:
        birefnet = None
        torch_gc()
        birefnet = BiRefNetPipeline(model_name)
    return birefnet


class ScriptPostprocessingBiRefNet(scripts_postprocessing.ScriptPostprocessing):
    name = "BiRefNet"
    order = 20001
    model = None

    def ui(self):
        with InputAccordion(False, label="Remove background with BiRefNet") as enable:
            with FormRow():
                model = gr.Dropdown(label="Remove background model", choices=models, value="None", info="Choose a BiRefNet model. Each model gives a different result.")
                resolution = gr.Textbox(label="Resolution", value="", placeholder="1024x1024", info="If left empty, it will take image size rounded to the nearest multiple of 32.")

            with FormRow():
                return_original = gr.Checkbox(label="Return original image")
                return_foreground = gr.Checkbox(True, label="Return foreground")
                return_mask = gr.Checkbox(label="Return mask")
                return_edge_mask = gr.Checkbox(label="Return edge mask")

            with FormRow(visible=False) as edge_mask_row:
                edge_mask_width = gr.Slider(label="Edge mask width", minimum=1, maximum=256, step=1, value=64)

            return_edge_mask.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[return_edge_mask],
                outputs=[edge_mask_row],
            )

        return {
            "enable": enable,
            "model": model,
            "resolution": resolution,
            "return_original": return_original,
            "return_foreground": return_foreground,
            "return_mask": return_mask,
            "return_edge_mask": return_edge_mask,
            "edge_mask_width": edge_mask_width,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, model, resolution, return_original, return_foreground, return_mask, return_edge_mask, edge_mask_width):
        if not enable:
            return

        if not model or model == "None":
            return
        
        birefnet = get_pipeline_using_cache(model)

        mask, foreground, edge_mask = birefnet.process(
            pp.image.convert("RGB"),
            resolution=resolution,
            return_mask=return_mask,
            return_foreground=return_foreground,
            return_edge_mask=return_edge_mask,
            edge_mask_width=edge_mask_width
        )

        if return_original:
            if foreground:
                pp.extra_images.append(foreground)
            if mask:
                pp.extra_images.append(mask)
            if edge_mask:
                pp.extra_images.append(edge_mask)
        else:
            output_image = foreground or mask or edge_mask
            if output_image:
                pp.image = output_image
                if foreground and foreground != output_image:
                    pp.extra_images.append(foreground)
                if mask and mask != output_image:
                    pp.extra_images.append(mask)
                if edge_mask and edge_mask != output_image:
                    pp.extra_images.append(edge_mask)

        pp.info["BiRefNet"] = model