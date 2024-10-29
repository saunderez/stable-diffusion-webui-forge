import pytest
from fastapi.testclient import TestClient
from modules.api.api import Api
from modules.api.models import StableDiffusionTxt2ImgProcessingAPI, StableDiffusionImg2ImgProcessingAPI, ExtrasSingleImageRequest, ExtrasBatchImagesRequest, PNGInfoRequest, InterrogateRequest
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules import shared, scripts, sd_samplers, postprocessing, images, infotext_utils, sd_models, deepbooru, devices
from modules.realesrgan_model import get_realesrgan_models
from modules.textual_inversion.textual_inversion import create_embedding
from modules.progress import create_task_id, add_task_to_queue, start_task, finish_task, current_task
from PIL import PngImagePlugin
from io import BytesIO
import base64
import os
import time
import datetime
import requests
import gradio as gr
from threading import Lock
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from secrets import compare_digest
from contextlib import closing
import piexif
import piexif.helper
import backend.memory_management as memory_management
import backend.attention as attention
from backend.diffusion_engine.base import ForgeDiffusionEngine
from backend.diffusion_engine.sd15 import StableDiffusion
from backend.diffusion_engine.sd20 import StableDiffusion2
from backend.diffusion_engine.sd35 import StableDiffusion3
from backend.diffusion_engine.sdxl import StableDiffusionXL
from backend.diffusion_engine.flux import Flux
from backend.loader import load_huggingface_component

app = FastAPI()
queue_lock = Lock()
api = Api(app, queue_lock)
client = TestClient(app)

def test_text2imgapi():
    request_data = {
        "prompt": "A beautiful landscape",
        "steps": 20,
        "sampler_name": "Euler a",
        "cfg_scale": 7.5,
        "seed": 42,
        "width": 512,
        "height": 512,
        "script_name": "",
        "script_args": [],
        "alwayson_scripts": {},
        "infotext": "",
        "send_images": True,
        "save_images": False
    }
    response = client.post("/sdapi/v1/txt2img", json=request_data)
    assert response.status_code == 200
    assert "images" in response.json()

def test_img2imgapi():
    init_image = images.read(BytesIO(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/6WRC5kAAAAASUVORK5CYII=")))
    init_image_b64 = base64.b64encode(init_image.tobytes()).decode("utf-8")
    request_data = {
        "init_images": [init_image_b64],
        "prompt": "A beautiful landscape",
        "steps": 20,
        "sampler_name": "Euler a",
        "cfg_scale": 7.5,
        "seed": 42,
        "width": 512,
        "height": 512,
        "script_name": "",
        "script_args": [],
        "alwayson_scripts": {},
        "infotext": "",
        "send_images": True,
        "save_images": False
    }
    response = client.post("/sdapi/v1/img2img", json=request_data)
    assert response.status_code == 200
    assert "images" in response.json()

def test_extras_single_image_api():
    image = images.read(BytesIO(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/6WRC5kAAAAASUVORK5CYII=")))
    image_b64 = base64.b64encode(image.tobytes()).decode("utf-8")
    request_data = {
        "image": image_b64,
        "upscaler_1": "None",
        "upscaler_2": "None",
        "extras_upscaler_1": "None",
        "extras_upscaler_2": "None",
        "resize_mode": 0,
        "show_extras_results": False,
        "gfpgan_visibility": 0,
        "codeformer_visibility": 0,
        "codeformer_weight": 0,
        "upscale_first": False
    }
    response = client.post("/sdapi/v1/extra-single-image", json=request_data)
    assert response.status_code == 200
    assert "image" in response.json()

def test_extras_batch_images_api():
    image = images.read(BytesIO(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/6WRC5kAAAAASUVORK5CYII=")))
    image_b64 = base64.b64encode(image.tobytes()).decode("utf-8")
    request_data = {
        "imageList": [{"data": image_b64}],
        "upscaler_1": "None",
        "upscaler_2": "None",
        "extras_upscaler_1": "None",
        "extras_upscaler_2": "None",
        "resize_mode": 0,
        "show_extras_results": False,
        "gfpgan_visibility": 0,
        "codeformer_visibility": 0,
        "codeformer_weight": 0,
        "upscale_first": False
    }
    response = client.post("/sdapi/v1/extra-batch-images", json=request_data)
    assert response.status_code == 200
    assert "images" in response.json()

def test_pnginfoapi():
    image = images.read(BytesIO(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/6WRC5kAAAAASUVORK5CYII=")))
    image_b64 = base64.b64encode(image.tobytes()).decode("utf-8")
    request_data = {
        "image": image_b64
    }
    response = client.post("/sdapi/v1/png-info", json=request_data)
    assert response.status_code == 200
    assert "info" in response.json()

def test_interrogateapi():
    image = images.read(BytesIO(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/6WRC5kAAAAASUVORK5CYII=")))
    image_b64 = base64.b64encode(image.tobytes()).decode("utf-8")
    request_data = {
        "image": image_b64,
        "model": "clip"
    }
    response = client.post("/sdapi/v1/interrogate", json=request_data)
    assert response.status_code == 200
    assert "caption" in response.json()
