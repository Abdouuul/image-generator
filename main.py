from diffusers import StableDiffusionPipeline
from flask import Flask, Response, request
import torch, io

app = Flask(__name__)

# Load model (first time download may be large ~4GB)
pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float16,
    safety_checker=None  
).to("cuda")

@app.route('/generate/<prompt>', methods= ['GET', 'POST'])
def generate_image(prompt):
    # encode the prompt to remove %20 type
    prompt = str(prompt).replace('%20', ' ')
    if request.method == 'POST':
        try:
            print('prompt recieved : ', str(prompt).strip())
            image = pipe(str(prompt).strip()).images[0]
            # image.save("output.png")
            img_io = io.BytesIO()
            image.save(img_io, format='PNG')
            img_io.seek(0)

            return Response(img_io.getvalue(), mimetype='image/png')
        except Exception as e :
            print(e)

# while True:
#     prompt = input('What would you like to create? \n')
#     image = pipe(prompt).images[0]

#     image.save("output.png")
#     image.show()


if __name__ == '__main__':
    app.run(debug=True)
