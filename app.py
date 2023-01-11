import argparse
import binascii
import glob
import openai
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import sys
import tempfile
import torch
from diffusers import StableDiffusionPipeline
from flask import Flask, redirect, render_template, request, url_for, send_file, send_from_directory


app = Flask(__name__)
app.config['IMAGE_EXTS'] = [".png", ".jpg", ".jpeg", ".gif", ".tiff"]
openai.api_key = os.getenv("OPENAI_API_KEY")

def encode(x):
    return binascii.hexlify(x.encode('utf-8')).decode()

def decode(x):
    return binascii.unhexlify(x.encode('utf-8')).decode()

@app.route("/", methods=("GET", "POST"))
def index():
    return render_template("index.html")

@app.route("/petnames", methods=("GET", "POST"))
def petnames():
    if request.method == "POST":
        animal = request.form["animal"]
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=generate_prompt(animal),
            temperature=0.6,
        )
        return redirect(url_for("petnames", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("petnames.html", result=result)

@app.route("/draw", methods=("GET", "POST"))
def draw():
    if request.method == "POST":
        drawing = request.form["drawing"]
        #pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        #pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5")
        pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5", torch_dtype=torch.float16)
        #pipe = StableDiffusionPipeline.from_pretrained("/tmp/stable-diffusion-v1-5", torch_dtype=torch.float16)
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
        artist_model_version = 'v1.5'
        drawing = request.form["drawing"]
        print("generating drawing '", drawing, "'")
        #drawing_filename = 'images/' + drawing.replace(' ', '_') + artist_model_version + '.png'
        drawing_filename = 'images/' + drawing.replace(' ', '_') + '.png'
        if os.path.exists(drawing_filename):
            return send_file(drawing_filename, mimetype='image/png')
        #pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        #pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5")
        #prompt = "a photo of an astronaut riding a dragon in paris"
        #image = pipe(prompt).images[0]  
        #image.save("astronaut_rides_horse.png")
        #image = pipe(drawing).images[0]  
        #return send_file(image, mimetype='image/png')
        pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5", torch_dtype=torch.float16)
        #pipe = StableDiffusionPipeline.from_pretrained("/tmp/stable-diffusion-v1-5", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        image = pipe(drawing).images[0]  
        image.seek(0)
        image.save(drawing_filename)
        return send_file(drawing_filename, mimetype='image/png')

# This next section was borrowed from: https://github.com/piyush01123/Flask-Image-Gallery
@app.route('/gallery')
def gallery():
    #root_dir = app.config['ROOT_DIR']
    root_dir = 'images'
    image_paths = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in app.config['IMAGE_EXTS']):
                image_paths.append(encode(os.path.join(root,file)))
    return render_template('gallery.html', paths=image_paths)

@app.route('/gallery_names')
def gallery_names():
    root_dir = 'images'
    file_arr = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in app.config['IMAGE_EXTS']):
                file_arr.append({ 'encoded_path': encode(os.path.join(root,file)), 'name': os.path.join(file)})
    return render_template('gallery_names.html', files=file_arr)

@app.route("/draw2", methods=("GET", "POST"))
def draw2():
    result = request.args.get("result")
    return render_template("draw2.html", result=result)

@app.route("/image2", methods=("GET", "POST"))
def image2():
    if request.method == "POST":
        drawing = request.form["drawing"]
        print("generating drawing '", drawing, "'")
        drawing_filename = 'images2/' + drawing.replace(' ', '_') + '.png'
        if os.path.exists(drawing_filename):
            return send_file(drawing_filename, mimetype='image/png')
        pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-2", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        image = pipe(drawing).images[0]  
        image.seek(0)
        image.save(drawing_filename)
        return send_file(drawing_filename, mimetype='image/png')

# This next section was borrowed from: https://github.com/piyush01123/Flask-Image-Gallery
@app.route('/gallery2')
def gallery2():
    #root_dir = app.config['ROOT_DIR']
    root_dir = 'images2'
    image_paths = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in app.config['IMAGE_EXTS']):
                image_paths.append(encode(os.path.join(root,file)))
    return render_template('gallery.html', paths=image_paths)

@app.route('/gallery2_names')
def gallery2_names():
    root_dir = 'images2'
    file_arr = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in app.config['IMAGE_EXTS']):
                file_arr.append({ 'encoded_path': encode(os.path.join(root,file)), 'name': os.path.join(file)})
    return render_template('gallery_names.html', files=file_arr)

# This next section was borrowed from: https://github.com/piyush01123/Flask-Image-Gallery
@app.route('/gallery2_1')
def gallery2_1():
    #root_dir = app.config['ROOT_DIR']
    root_dir = 'images2-1'
    image_paths = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in app.config['IMAGE_EXTS']):
                image_paths.append(encode(os.path.join(root,file)))
    return render_template('gallery.html', paths=image_paths)

@app.route('/gallery2_1_names')
def gallery2_1_names():
    root_dir = 'images2-1'
    file_arr = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in app.config['IMAGE_EXTS']):
                file_arr.append({ 'encoded_path': encode(os.path.join(root,file)), 'name': os.path.join(file)})
    return render_template('gallery_names.html', files=file_arr)

@app.route('/cdn/<path:filepath>')
def download_file(filepath):
    dir,filename = os.path.split(decode(filepath))
    return send_from_directory(dir, filename, as_attachment=False)
if __name__=="__main__":
    parser = argparse.ArgumentParser('Usage: %prog [options]')
    parser.add_argument('root_dir', help='Gallery root directory path')
    parser.add_argument('-l', '--listen', dest='host', default='127.0.0.1', \
                                    help='address to listen on [127.0.0.1]')
    parser.add_argument('-p', '--port', metavar='PORT', dest='port', type=int, \
                                default=5000, help='port to listen on [5000]')
    args = parser.parse_args()
    app.config['ROOT_DIR'] = args.root_dir
    app.run(host=args.host, port=args.port, debug=True)

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
