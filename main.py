from diffusers import StableDiffusionPipeline
import torch

# Load model (first time download may be large ~4GB)
pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float16,
).to("cuda")

while True:
    prompt = input('What would you like to create? \n')
    image = pipe(prompt).images[0]

    image.save("output.png")
    image.show()