from modules import scripts_postprocessing, ui_components
import gradio as gr

from modules.ui_components import FormRow
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


class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "BiRefNet"
    order = 20001
    model = None

    def ui(self):
        with ui_components.InputAccordion(False, label="Remove background with BiRefNet") as enable:
            with FormRow():
                model = gr.Dropdown(label="Remove background", choices=models, value="None")
                return_foreground = gr.Checkbox(label="Return foreground", value=False)
                return_edge_mask = gr.Checkbox(label="Return edge mask", value=False)

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
            "return_foreground": return_foreground,
            "return_edge_mask": return_edge_mask,
            "edge_mask_width": edge_mask_width,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, model, return_foreground, return_edge_mask, edge_mask_width):
        if not enable:
            return

        if not model or model == "None":
            return
        
        birefnet = get_pipeline_using_cache(model)

        mask, output_image, edge_mask = birefnet.process(
            pp.image.convert("RGB"),
            resolution='',
            return_foreground=return_foreground,
            return_edge_mask=return_edge_mask,
            edge_mask_width=edge_mask_width
        )

        pp.image = output_image
        if mask:
            pp.extra_images.append(mask)
        if edge_mask:
            pp.extra_images.append(edge_mask)

        pp.info["BiRefNet"] = model