def test_inversion_ddim():
    from inversion_edits import ddim_inversion_latent, ddim_first_skipped_latent

    from diffusers import DDIMScheduler, StableDiffusionPipeline
    import torch

    model_id = "runwayml/stable-diffusion-v1-5"
    # model_id = "./cat_finetuned"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda:0")

    import PIL

    x = PIL.Image.open("example_scripts/horse.jpg")
    lat = ddim_first_skipped_latent(
        pipe, x, condition_prompt="", num_inference_steps=50, skip_ratio=0.5
    )

    pipe.safety_checker = None
    x_reco = pipe(prompt="", latents=lat)
    x_reco.images[0].save("example_scripts/horse_reco.jpg")


if __name__ == "__main__":
    test_inversion_ddim()
