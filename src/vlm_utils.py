import math

import torch
import torchvision.transforms as T
import numpy as np
#from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.95,
}


def build_generation_config(tokenizer):
    """Return the shared InternVL generation settings for this project."""
    return {
        **DEFAULT_GENERATION_CONFIG,
        "pad_token_id": tokenizer.eos_token_id,
    }

def build_transform(input_size):
    
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def build_internvl_device_map(model_name, world_size=None):
    """Build InternVL's recommended asymmetric map for multi-GPU inference."""
    if world_size is None:
        world_size = torch.cuda.device_count()
    if world_size < 2:
        return None

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers

    # The vision encoder is placed on GPU 0, so InternVL treats that GPU as
    # having half the capacity available for language-model layers.
    layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    layer_allocations = [layers_per_gpu] * world_size
    layer_allocations[0] = math.ceil(layers_per_gpu * 0.5)

    device_map = {}
    layer_index = 0
    for gpu_index, allocation in enumerate(layer_allocations):
        for _ in range(allocation):
            if layer_index >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_index}"] = gpu_index
            layer_index += 1

    device_map.update({
        "vision_model": 0,
        "mlp1": 0,
        "language_model.model.embed_tokens": 0,
        "language_model.model.norm": 0,
        "language_model.model.rotary_emb": 0,
        "language_model.lm_head": 0,
    })
    # InternVL requires the first and last language layers on the same GPU.
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


def load_model(model_name, quant=True):
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "use_flash_attn": True,
        "trust_remote_code": True,
    }
    if quant:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    device_map = build_internvl_device_map(model_name)
    if device_map is not None:
        model_kwargs["device_map"] = device_map
        print(f"Loading {model_name} across {torch.cuda.device_count()} GPUs.")

    model = AutoModel.from_pretrained(model_name, **model_kwargs).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    
    return model, tokenizer


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
