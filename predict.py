# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import mimetypes
import json
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper
from comfyui_enums import SAMPLERS, SCHEDULERS

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

api_json_file = "workflow_api.json"

SD3_MODELS = [
    "sd3_medium_incl_clips.safetensors",
    "sd3_medium_incl_clips_t5xxlfp16.safetensors",
    "sd3_medium_incl_clips_t5xxlfp8.safetensors",
]


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        pass

    def update_workflow(self, workflow, **kwargs):
        checkpoint_loader = workflow["252"]["inputs"]
        checkpoint_loader["ckpt_name"] = kwargs["model"]

        shift = workflow["13"]["inputs"]
        shift["shift"] = kwargs["shift"]

        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["289"]["inputs"]
        negative_prompt["text"] = kwargs["negative_prompt"]

        sampler = workflow["271"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["cfg"] = kwargs["guidance_scale"]
        sampler["scheduler"] = kwargs["scheduler"]
        sampler["sampler_name"] = kwargs["sampler"]
        sampler["steps"] = kwargs["steps"]

        negative_conditioning = workflow["274"]["inputs"]
        negative_conditioning["end"] = kwargs["negative_conditioning_end"]

        empty_latent_image = workflow["275"]["inputs"]
        empty_latent_image["width"] = kwargs["width"]
        empty_latent_image["height"] = kwargs["height"]
        empty_latent_image["batch_size"] = kwargs["number_of_images"]

        if kwargs["use_triple_prompt"]:
            triple_prompt = workflow["291"]["inputs"]
            triple_prompt["clip_g"] = kwargs["triple_prompt_clip_g"]
            triple_prompt["clip_l"] = kwargs["triple_prompt_clip_l"]
            triple_prompt["t5xxl"] = kwargs["triple_prompt_t5"]
            triple_prompt["empty_padding"] = (
                "empty_prompt" if kwargs["triple_prompt_empty_padding"] else "none"
            )
        else:
            del workflow["291"]
            sampler["positive"] = ["6", 0]

    def predict(
        self,
        prompt: str = Input(
            default="",
            description="This prompt is ignored when using the triple prompt mode. See below.",
        ),
        model: str = Input(
            choices=SD3_MODELS,
            default="sd3_medium_incl_clips_t5xxlfp16.safetensors",
        ),
        width: int = Input(
            description="The width of the image",
            default=1024,
        ),
        height: int = Input(
            description="The height of the image",
            default=1024,
        ),
        steps: int = Input(
            description="The number of steps to run the diffusion model for",
            default=28,
        ),
        sampler: str = Input(
            description="The sampler to use for the diffusion model",
            choices=SAMPLERS,
            default="dpmpp_2m",
        ),
        scheduler: str = Input(
            description="The scheduler to use for the diffusion model",
            choices=SCHEDULERS,
            default="sgm_uniform",
        ),
        shift: float = Input(
            description="The timestep scheduling shift. Try values 6.0 and 2.0 to experiment with effects.",
            le=20,
            ge=0,
            default=3.0,
        ),
        guidance_scale: float = Input(
            description="The guidance scale tells the model how similar the output should be to the prompt.",
            le=20,
            ge=0,
            default=4.5,
        ),
        number_of_images: int = Input(
            description="The number of images to generate",
            le=10,
            ge=1,
            default=1,
        ),
        use_triple_prompt: bool = Input(
            default=False,
        ),
        triple_prompt_clip_g: str = Input(
            description="The prompt that will be passed to just the CLIP-G model.",
            default="",
        ),
        triple_prompt_clip_l: str = Input(
            description="The prompt that will be passed to just the CLIP-L model.",
            default="",
        ),
        triple_prompt_t5: str = Input(
            description="The prompt that will be passed to just the T5-XXL model.",
            default="",
        ),
        triple_prompt_empty_padding: bool = Input(
            description="Whether to add padding for empty prompts. Useful if you only want to pass a prompt to one or two of the three text encoders. Has no effect when all prompts are filled. Disable this for interesting effects.",
            default=True,
        ),
        negative_prompt: str = Input(
            description="Negative prompts do not really work in SD3. This will simply cause your output image to vary in unpredictable ways.",
            default="",
        ),
        negative_conditioning_end: float = Input(
            description="When the negative conditioning should stop being applied. By default it is disabled.",
            le=20,
            ge=0,
            default=0,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        if (
            use_triple_prompt
            and triple_prompt_t5
            and model == "sd3_medium_incl_clips.safetensors"
        ):
            print(
                "WARNING: The T5 prompt will be ignored because the sd3_medium_incl_clips.safetensors model does not include a T5 encoder"
            )

        self.update_workflow(
            workflow,
            prompt=prompt,
            model=model,
            seed=seed,
            width=width,
            height=height,
            shift=shift,
            steps=steps,
            sampler=sampler,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            number_of_images=number_of_images,
            use_triple_prompt=use_triple_prompt,
            triple_prompt_clip_g=triple_prompt_clip_g,
            triple_prompt_clip_l=triple_prompt_clip_l,
            triple_prompt_t5=triple_prompt_t5,
            triple_prompt_empty_padding=triple_prompt_empty_padding,
            negative_prompt=negative_prompt,
            negative_conditioning_end=negative_conditioning_end,
        )

        self.comfyUI.connect()
        self.comfyUI.run_workflow(workflow)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
