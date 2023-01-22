# Author : Simo Ryu

# Referenced
# https://github.com/cccntu/efficient-prompt-to-prompt/blob/main/ddim-inversion.ipynb
# https://arxiv.org/pdf/2105.05233.pdf
# https://arxiv.org/pdf/2211.09794.pdf
# https://arxiv.org/pdf/2211.12572.pdf

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)
from PIL import Image
from tqdm.auto import tqdm


def _preprocess(image: PIL.Image):
    w, h = 512, 512
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def _backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    """from noise to image"""
    # From https://github.com/cccntu/efficient-prompt-to-prompt/blob/main/ddim-inversion.ipynb.
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )


@torch.no_grad()
def ddim_inversion_latent(
    pipe: StableDiffusionPipeline,
    image: PIL.Image,
    num_inference_steps: int = 50,
    condition_prompt: str = "",
):
    if not isinstance(pipe.scheduler, DDIMScheduler):
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        print("Scheduler for Pipe is now DDIMScheduler")

    vae, unet, text_encoder, tokenizer, scheduler = (
        pipe.vae,
        pipe.unet,
        pipe.text_encoder,
        pipe.tokenizer,
        pipe.scheduler,
    )

    device = pipe.device
    dtype = pipe.unet.dtype

    text_input = tokenizer(
        [condition_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    cond_emb = text_encoder(text_input.input_ids.to(device))[0]

    image = _preprocess(image)
    image = image.to(device=device).to(dtype)
    x_T = vae.encode(image).latent_dist.sample()
    x_T = 0.18215 * x_T

    latents = x_T.clone()

    scheduler.set_timesteps(num_inference_steps)
    reverse_process = True
    for t in tqdm(reversed(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = latents

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        noise_pred = unet(latent_model_input, t, encoder_hidden_states=cond_emb).sample

        prev_timestep = (
            t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        )

        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else scheduler.final_alpha_cumprod
        )

        latents = _backward_ddim(
            x_t=latents,
            alpha_t=alpha_prod_t_prev,
            alpha_tm1=alpha_prod_t,
            eps_xt=noise_pred,
        )

    return latents
