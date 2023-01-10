import argparse
import binascii
import glob
import openai
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import random
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

def fake_gan():
    images = [
        (random.choice(
            [
                "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80",
                "https://images.unsplash.com/photo-1554151228-14d9def656e4?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=386&q=80",
                "https://images.unsplash.com/photo-1542909168-82c3e7fdca5c?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8aHVtYW4lMjBmYWNlfGVufDB8fDB8fA%3D%3D&w=1000&q=80",
                "https://images.unsplash.com/photo-1546456073-92b9f0a8d413?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80",
                "https://images.unsplash.com/photo-1601412436009-d964bd02edbc?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=464&q=80",
            ]
        ), f"label {i}" if i != 0 else "label" * 50)
        for i in range(3)
    ]
    return images

def draw(inp, this_model, force_new):
    drawing = inp
    if this_model == "stable-diffusion-2":
        this_model_addr = "../stable-diffusion-2"
        images_dir = 'images2/'
    elif this_model == "stable-diffusion-2-1":
        this_model_addr = "../stable-diffusion-2-1"
        images_dir = 'images2-1/'
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
    pipe.enable_attention_slicing()
    pipe = pipe.to("cuda")
    image = pipe(drawing).images[0]  
    image.seek(0)
    image.save(drawing_filename)
    return image

drawdemo = gr.Interface(
    fn=draw,
    inputs=[
        gr.Text(label="Drawing description text", placeholder="astronaut riding a horse on mars"),
        gr.Dropdown(label='Model', choices=["stable-diffusion-2", "stable-diffusion-2-1", "stable-diffusion-v1-5"], value="stable-diffusion-2"),
        gr.Checkbox(label="Force-New"),
    ],
    outputs="image",
        examples=[
        ['van gogh dogs playing poker', "stable-diffusion-v1-5", False],
        ['picasso dogs playing poker', "stable-diffusion-v1-5", False],
        ['dali dogs playing poker', "stable-diffusion-v1-5", False],
        ['matisse dogs playing poker', "stable-diffusion-v1-5", False],
        ['manet dogs playing poker', "stable-diffusion-v1-5", False],
        ['monet dogs playing poker', "stable-diffusion-v1-5", False],
        ['hindu mandala copper and patina green', "stable-diffusion-v1-5", False],
        ['hindu mandala fruit salad', "stable-diffusion-v1-5", False],
        ['hindu mandala psychedelic', "stable-diffusion-v1-5", False],
        ['astronaut riding a horse on mars', "stable-diffusion-v1-5", False],
        ['astronaut riding a horse on mars', "stable-diffusion-2", False],
        ['astronaut riding a horse on mars', "stable-diffusion-2-1", False]
    ],
)

with gr.Blocks() as gallerydemo:
    with gr.Column(variant="panel"):
        with gr.Row(variant="compact"):
            text = gr.Textbox(
                label="Enter your prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
            ).style(
                container=False,
            )
            btn = gr.Button("Generate image").style(full_width=False)

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

    btn.click(fake_gan, None, gallery)

demo = gr.TabbedInterface( [drawdemo, gallerydemo], ["Draw", "Gallery"])
demo.queue(
    concurrency_count = 1,
    max_size = 4
)
print(SERVER_NAME)
demo.launch(
    server_name = SERVER_NAME,
    max_threads = 1,
    enable_queue = True
)
