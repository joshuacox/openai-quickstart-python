import os
import numpy as np
import torch
import tempfile
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline


import openai
from flask import Flask, redirect, render_template, request, url_for, send_file

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        animal = request.form["animal"]
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=generate_prompt(animal),
            temperature=0.6,
        )
        return redirect(url_for("index", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("index.html", result=result)

@app.route("/draw", methods=("GET", "POST"))
def draw():
    if request.method == "POST":
        drawing = request.form["drawing"]
        #pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        #pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5")
        pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        #prompt = "a photo of an astronaut riding a dragon in paris"
        #image = pipe(prompt).images[0]  
        #image.save("astronaut_rides_horse.png")
        image = pipe(drawing).images[0]  
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            plt.imsave(f, image, cmap='gray')
            f.seek(0)
        return send_file(f, mimetype='image/png')
        #return redirect(url_for("draw", image=f))

    result = request.args.get("result")
    return render_template("draw.html", result=result)

@app.route("/image", methods=("GET", "POST"))
def image():
    if request.method == "POST":
        drawing = request.form["drawing"]
        #pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        #pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5")
        #prompt = "a photo of an astronaut riding a dragon in paris"
        #image = pipe(prompt).images[0]  
        #image.save("astronaut_rides_horse.png")
        #image = pipe(drawing).images[0]  
        #return send_file(image, mimetype='image/png')
        pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        image = pipe(drawing).images[0]  
        image.seek(0)
        image.save('/tmp/tmp.png')
        #f = pipe(drawing).images[0]  
        return send_file('/tmp/tmp.png', mimetype='image/png')

def bork():
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5", torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            #plt.imsave(f, image)
            image = pipe(drawing).images[0]  
            image.seek(0)
            image.save(f.name)
            #f = pipe(drawing).images[0]  
            return send_file(f, mimetype='image/png')


def generate_prompt(animal):
    return """Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(
        animal.capitalize()
    )

def generate_prompt_drawing(drawing):
    return """Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(
        drawing.capitalize()
    )
