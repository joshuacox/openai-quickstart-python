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
from PIL import Image
from diffusers import StableDiffusionPipeline

import gradio as gr

artist_model_version = 'v2'

def draw(inp):
    drawing = inp
    drawing_filename = 'images2/' + drawing.replace(' ', '_') + '.png'
    if os.path.exists(drawing_filename):
        print("found drawing ", drawing_filename)
        return Image.open(drawing_filename) 
    print("generating drawing '", drawing, "'", drawing_filename)
    #pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-2", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(drawing).images[0]  
    image.seek(0)
    image.save(drawing_filename)
    return image

#demo = gr.Interface(fn=draw, inputs="text", outputs="image", flask_host_name="10.11.5.11")
demo = gr.Interface(fn=draw, inputs="text", outputs="image")
demo.launch(server_name="10.11.5.11")
