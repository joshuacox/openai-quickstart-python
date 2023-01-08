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

def image_classifier(inp):
    return {'cat': 0.3, 'dog': 0.7}


def draw(inp):
    drawing = inp
    drawing_filename = 'images/' + drawing.replace(' ', '_') + '.png'
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

demo = gr.Interface(fn=draw, inputs="text", outputs="image")
demo.launch()
