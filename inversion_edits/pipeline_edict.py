from diffusers import StableDiffusionPipeline, DDIMScheduler

import numpy as np
import torch
import PIL
from tqdm import tqdm


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h)))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


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

    return sa * ((1 / sb) * x_tm1 + ((1 / a - 1) ** 0.5 - (1 / b - 1) ** 0.5) * eps_xt)


class EDICTPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def edict_inversion_latent(
        self,
        image: PIL.Image,
        num_inference_steps: int = 50,
        condition_prompt: str = "",
    ):
        self.ddim_scheduler = DDIMScheduler.from_config(self.scheduler.config)
        text_input = self.tokenizer(
            [condition_prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        device = self.device
        dtype = self.unet.dtype

        cond_emb = self.text_encoder(text_input.input_ids.to(device))[0]

        image = preprocess(image)
        image = image.to(device=device).to(dtype)
        x_T = self.vae.encode(image).latent_dist.sample()
        x_T = 0.18215 * x_T

        latents = x_T.clone()

        self.ddim_scheduler.set_timesteps(num_inference_steps)
        prev_timestep = None

        for t in tqdm(reversed(self.ddim_scheduler.timesteps)):
            latent_model_input = latents
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=cond_emb
            ).sample

            alpha_prod_t = self.ddim_scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.ddim_scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep is not None
                else self.ddim_scheduler.final_alpha_cumprod
            )
            prev_timestep = t

            latents = _backward_ddim(
                x_tm1=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )

        return latents
