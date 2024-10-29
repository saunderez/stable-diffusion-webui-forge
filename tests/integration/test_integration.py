import pytest
from fastapi.testclient import TestClient
from modules.api.api import Api
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules import shared, scripts, sd_samplers, sd_models
from modules.api.models import StableDiffusionTxt2ImgProcessingAPI, StableDiffusionImg2ImgProcessingAPI
from backend.loader import load_huggingface_component
from backend.memory_management import get_torch_device, get_total_memory, get_free_memory, load_models_gpu, load_model_gpu, unload_all_models, soft_empty_cache, xformers_enabled, xformers_enabled_vae, pytorch_attention_enabled, force_upcast_attention_dtype, cast_to_device, should_use_fp16, should_use_bf16, unet_offload_device, unet_inital_load_device, unet_dtype, get_computation_dtype, text_encoder_offload_device, text_encoder_device, text_encoder_dtype, vae_device, vae_offload_device, vae_dtype
from backend.attention import get_attn_precision, attention_basic, attention_sub_quad, attention_split, attention_xformers, attention_pytorch, slice_attention_single_head_spatial, normal_attention_single_head_spatial, xformers_attention_single_head_spatial, pytorch_attention_single_head_spatial, AttentionProcessorForge

app = FastAPI()
api = Api(app, Lock())
client = TestClient(app)

def test_txt2img_api():
    txt2imgreq = StableDiffusionTxt2ImgProcessingAPI(
        prompt="A beautiful landscape",
        steps=20,
        sampler_name="Euler a",
        cfg_scale=7.5,
        seed=42,
        width=512,
        height=512,
        batch_size=1,
        n_iter=1,
        save_images=False,
        send_images=True,
    )
    response = client.post("/sdapi/v1/txt2img", json=txt2imgreq.dict())
    assert response.status_code == 200
    data = response.json()
    assert "images" in data
    assert len(data["images"]) == 1

def test_img2img_api():
    img2imgreq = StableDiffusionImg2ImgProcessingAPI(
        init_images=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAgAB/ax5LIAAAAASUVORK5CYII="],
        prompt="A beautiful landscape",
        steps=20,
        sampler_name="Euler a",
        cfg_scale=7.5,
        seed=42,
        width=512,
        height=512,
        batch_size=1,
        n_iter=1,
        save_images=False,
        send_images=True,
    )
    response = client.post("/sdapi/v1/img2img", json=img2imgreq.dict())
    assert response.status_code == 200
    data = response.json()
    assert "images" in data
    assert len(data["images"]) == 1
