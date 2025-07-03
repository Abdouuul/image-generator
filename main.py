from diffusers import StableDiffusionPipeline
import torch

# Load model (first time download may be large ~4GB)
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = input('What would you like to create? \n')
image = pipe(prompt).images[0]

image.save("output.png")