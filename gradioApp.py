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

import gradio as gr

def image_classifier(inp):
    return {'cat': 0.3, 'dog': 0.7}


def draw(inp):
    drawing = inp
    pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(drawing).images[0]  
    return image

demo = gr.Interface(fn=draw, inputs="text", outputs="image")
demo.launch()
