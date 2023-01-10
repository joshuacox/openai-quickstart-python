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
import time
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

import gradio as gr

from dotenv import load_dotenv
load_dotenv()
SERVER_NAME = os.getenv("SERVER_NAME")


def draw(inp, this_model, force_new):
    drawing = inp
    if this_model == "stable-diffusion-2":
        this_model_addr = "../stable-diffusion-2"
        images_dir = 'images2/'
    elif this_model == "stable-diffusion-v1-5":
        this_model_addr = "../stable-diffusion-v1-5"
        images_dir = 'images/'
    else:
        raise gr.Error("Unknown Model!")
    drawing_filename = images_dir + drawing.replace(' ', '_') + '.png'
    if os.path.exists(drawing_filename):
        if force_new:
            new_drawing_filename = images_dir + drawing.replace(' ', '_') + '.' + str(time.time()) + '.png'
            os.replace(drawing_filename, new_drawing_filename)
        else:
            print("found drawing ", drawing_filename)
            return Image.open(drawing_filename) 
    print("generating drawing '", drawing, "'", drawing_filename)
    pipe = StableDiffusionPipeline.from_pretrained(this_model_addr, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(drawing).images[0]  
    image.seek(0)
    image.save(drawing_filename)
    return image

demo = gr.Interface(
    fn=draw,
    inputs=[
        gr.Text(label="Drawing description text", placeholder="astronaut riding a horse on mars"),
        gr.Dropdown(label='Model', choices=["stable-diffusion-2", "stable-diffusion-v1-5"], value="stable-diffusion-2"),
        gr.Checkbox(label="Force-New"),
    ],
    outputs="image",
        examples=[
        ['van gogh dogs playing poker', "stable-diffusion-2", False],
        ['astronaut riding a horse on mars', "stable-diffusion-2", False],
        ['astronaut riding a horse on mars', "stable-diffusion-v1-5", False]
    ],
)
demo.queue(
    concurrency_count = 1,
    max_size = 8
)
print(SERVER_NAME)
demo.launch(
    server_name = SERVER_NAME,
    max_threads = 1,
    enable_queue = True
)
