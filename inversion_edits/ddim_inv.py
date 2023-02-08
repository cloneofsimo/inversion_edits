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


def _backward_ddim(x_tm1, alpha_t, alpha_tm1, eps_xt):
    """
    let a = alpha_t, b = alpha_{t - 1}
    We have a > b,
    x_{t} - x_{t - 1} = sqrt(a) ((sqrt(1/b) - sqrt(1/a)) * x_{t-1} + (sqrt(1/a - 1) - sqrt(1/b - 1)) * eps_{t-1})
    From https://arxiv.org/pdf/2105.05233.pdf, section F.
    """

    a, b = alpha_t, alpha_tm1
    sa = a**0.5
    sb = b**0.5

    return (
        sa
        * (
            (1 / sb - 1 / sa) * x_tm1
            + ((1 / a - 1) ** 0.5 - (1 / b - 1) ** 0.5) * eps_xt
        )
        + x_tm1
    )


def _prepare(pipe, image, condition_prompt):
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
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

    return latents, cond_emb, scheduler


@torch.no_grad()
def ddim_inversion_latent(
    pipe: StableDiffusionPipeline,
    image: PIL.Image,
    num_inference_steps: int = 50,
    condition_prompt: str = "",
):
    latents, cond_emb, scheduler = _prepare(pipe, image, condition_prompt)

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(reversed(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = latents

        # latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=cond_emb
        ).sample

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
            x_tm1=latents,
            alpha_t=alpha_prod_t,
            alpha_tm1=alpha_prod_t_prev,
            eps_xt=noise_pred,
        )

    return latents


@torch.no_grad()
def ddim_first_skipped_latent(
    pipe: StableDiffusionPipeline,
    image: PIL.Image,
    num_inference_steps: int = 50,
    condition_prompt: str = "",
    skip_ratio: float = 0.5,
):
    latents, cond_emb, scheduler = _prepare(pipe, image, condition_prompt)
    total_steps = int(num_inference_steps * (1 + skip_ratio))
    initial_steps = total_steps - num_inference_steps

    scheduler.set_timesteps(total_steps)
    subsampled_schedule = list(scheduler.timesteps[::2])[:initial_steps] + list(
        scheduler.timesteps[2 * initial_steps :]
    )

    prev_timestep = None
    for idx, t in enumerate(tqdm(reversed(subsampled_schedule))):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = latents


        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=cond_emb
        ).sample

        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep
            else scheduler.final_alpha_cumprod
        )
        prev_timestep = t

        latents = _backward_ddim(
            x_tm1=latents,
            alpha_t=alpha_prod_t,
            alpha_tm1=alpha_prod_t_prev,
            eps_xt=noise_pred,
        )

    return latents


def edict_inversion_latent(
    pipe: StableDiffusionPipeline,
    image: PIL.Image,
    num_inference_steps: int = 50,
    condition_prompt: str = "",
):
    pass
