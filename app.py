<<<<<<< Updated upstream
import os
import numpy as np
import torch
import tempfile
import os.path
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline


=======
import argparse
import binascii
import glob
>>>>>>> Stashed changes
import openai
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import tempfile
import torch
#import os.path
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
        print("generating drawing '", drawing, "'")
        drawing_filename = 'images/' + drawing.replace(' ', '_') + '.png'
        if os.path.exists(drawing_filename):
            return send_file(drawing_filename, mimetype='image/png')
<<<<<<< Updated upstream
        #pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        #pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5")
        #prompt = "a photo of an astronaut riding a dragon in paris"
        #image = pipe(prompt).images[0]  
        #image.save("astronaut_rides_horse.png")
        #image = pipe(drawing).images[0]  
        #return send_file(image, mimetype='image/png')
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5", torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            image = pipe(drawing).images[0]  
            image.seek(0)
            #image.save('/tmp/tmp.png')
            #image.save(f.name)
            image.save(drawing_filename)
            #f = pipe(drawing).images[0]  
            #return send_file(f.name, mimetype='image/png')
            return send_file(drawing_filename, mimetype='image/png')
=======
        pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        image = pipe(drawing).images[0]  
        image.seek(0)
        image.save(drawing_filename)
        return send_file(drawing_filename, mimetype='image/png')
>>>>>>> Stashed changes

# This next section was borrowed from: https://github.com/piyush01123/Flask-Image-Gallery
@app.route('/gallery')
def home():
    #root_dir = app.config['ROOT_DIR']
    root_dir = 'images'
    image_paths = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in app.config['IMAGE_EXTS']):
                image_paths.append(encode(os.path.join(root,file)))
    return render_template('gallery.html', paths=image_paths)


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
