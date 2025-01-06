import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import gradio as gr  # Gradio 라이브러리 추가

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    if path is None:
        path = os.getcwd()

    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    parent_directory = os.path.dirname(path)

    if parent_directory == path:
        return None

    return find_path(name, parent_directory)

def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")

def add_extra_model_paths() -> None:
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")

add_comfyui_directory_to_sys_path()
add_extra_model_paths()

def import_custom_nodes() -> None:
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    init_extra_nodes()

from nodes import NODE_CLASS_MAPPINGS
from PIL import Image
import numpy as np

def tensor_to_pil(tensor):
    """
    Converts a PyTorch tensor to a PIL image.
    Args:
        tensor (torch.Tensor): The tensor to convert. Expected shape: (B, H, W, C) or (H, W, C).

    Returns:
        PIL.Image: Converted PIL image.
    """
    # Remove batch dimension if present
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Check for valid shape
    if tensor.shape[-1] != 3:
        raise ValueError(f"Unsupported tensor shape for RGB image conversion: {tensor.shape}")

    # Convert to NumPy array
    array = tensor.cpu().numpy()

    # Scale to [0, 255] and convert to uint8
    array = (array * 255).clip(0, 255).astype(np.uint8)

    # Convert to PIL image
    return Image.fromarray(array)

def process_image(image):
    import_custom_nodes()

    # Specify GPU device (change "cuda:0" to "cuda:1" if needed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def transform(image, seed):
        with torch.inference_mode():
            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            loadimage_10 = loadimage.load_image(image=image)

            checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            checkpointloadersimple_14 = checkpointloadersimple.load_checkpoint(
                ckpt_name="realcartoonPixar_v12.safetensors"
            )

            vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
            vaeencode_12 = vaeencode.encode(
                pixels=get_value_at_index(loadimage_10, 0).to(device),
                vae=get_value_at_index(checkpointloadersimple_14, 2),  # No .to(device) here
            )

            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            positive = cliptextencode.encode(
                text="Your default positive prompt",
                clip=get_value_at_index(checkpointloadersimple_14, 1),  # No .to(device) here
            )
            negative = cliptextencode.encode(
                text="Your default negative prompt",
                clip=get_value_at_index(checkpointloadersimple_14, 1),  # No .to(device) here
            )

            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

            ksampler_3 = ksampler.sample(
                seed=seed,
                steps=30,
                cfg=7,
                sampler_name="dpmpp_3m_sde_gpu",
                scheduler="karras",
                denoise=0.45,
                model=get_value_at_index(checkpointloadersimple_14, 0),  # No .to(device)
                positive=get_value_at_index(positive, 0) if positive else None,
                negative=get_value_at_index(negative, 0) if negative else None,
                latent_image=get_value_at_index(vaeencode_12, 0),
            )

            # Handle latent_image being a dictionary
            samples = get_value_at_index(ksampler_3, 0)
            if isinstance(samples, torch.Tensor):
                samples = samples.to(device)

            vaedecode_8 = vaedecode.decode(
                samples=samples,
                vae=get_value_at_index(checkpointloadersimple_14, 2),  # No .to(device) here
            )

            # Convert the output tensor to a PIL image
            output_tensor = get_value_at_index(vaedecode_8, 0)
            output_image = tensor_to_pil(output_tensor)
            return output_image

    # Generate two outputs with different seeds
    output_image1 = transform(image, seed=random.randint(1, 2**64))
    output_image2 = transform(image, seed=random.randint(1, 2**64))
    return output_image1, output_image2

# Gradio Interface
def main():
    interface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="filepath", label="Input Image"),
        outputs=[
            gr.Image(label="Output Image 1"),
            gr.Image(label="Output Image 2")
        ],
        title="Image Transformation with ComfyUI",
        description="Upload an image to generate two transformed outputs using the ComfyUI processing pipeline."
    )

    interface.launch(share=True)

if __name__ == "__main__":
    main()
