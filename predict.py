from cog import BasePredictor, Input, Path
import os
import torch
import time
import subprocess
import numpy as np
from typing import List
from diffusers import DiffusionPipeline, AutoencoderTiny, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from live_preview_helpers import flux_pipe_call_that_returns_an_iterable_of_images

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/ostris/Flex.1-alpha/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.dtype = torch.bfloat16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Download weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Load models
        self.taef1 = AutoencoderTiny.from_pretrained(
            "madebyollin/taef1", 
            torch_dtype=self.dtype
        ).to(self.device)
        
        self.good_vae = AutoencoderKL.from_pretrained(
            MODEL_CACHE, 
            subfolder="vae", 
            torch_dtype=self.dtype
        ).to(self.device)
        
        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_CACHE, 
            torch_dtype=self.dtype,
            vae=self.taef1,
        ).to(self.device)

        self.pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(self.pipe)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="an astronaut riding a horse on the moon"
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
            ge=256,
            le=2048,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
            ge=256,
            le=2048,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=28,
            ge=1,
            le=50,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=3.5,
            ge=1.0,
            le=15.0,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator().manual_seed(seed)

        # Get the last image from the generator
        output = None
        for image in self.pipe.flux_pipe_call_that_returns_an_iterable_of_images(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            output_type="pil",
            good_vae=self.good_vae,
        ):
            output = image

        output_path = Path(f"/tmp/output.png")
        output.save(output_path)
        return output_path 