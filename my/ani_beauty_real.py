import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import gradio as gr  # Gradio 라이브러리 추가
from PIL import Image
import numpy as np


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """Recursively searches parent folders for the given name."""
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
    """Add 'ComfyUI' to sys.path."""
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """Add paths from extra_model_paths.yaml to sys.path."""
    try:
        from main import load_extra_path_config
    except ImportError:
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Initialize custom nodes."""
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


def tensor_to_pil(tensor):
    """
    Converts a PyTorch tensor to a PIL image.
    """
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    if tensor.shape[-1] != 3:
        raise ValueError(f"Invalid shape for RGB conversion: {tensor.shape}")

    array = tensor.cpu().numpy()
    array = (array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def process_image(image):
    import_custom_nodes()
    with torch.inference_mode():
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loaded_image = loadimage.load_image(image=image)

        checkpointloader = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpoint = checkpointloader.load_checkpoint(
            ckpt_name="brmAnimeBeautyrealmix_v41.safetensors"
        )

        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        vae_encoded = vaeencode.encode(
            pixels=get_value_at_index(loaded_image, 0),
            vae=get_value_at_index(checkpoint, 2),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        positive = cliptextencode.encode(
            text="Generated description",
            clip=get_value_at_index(checkpoint, 1),
        )
        negative = cliptextencode.encode(
            text="Negative description",
            clip=get_value_at_index(checkpoint, 1),
        )

        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        sampled = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=30,
            cfg=7,
            sampler_name="euler_ancestral",
            scheduler="karras",
            denoise=0.45,
            model=get_value_at_index(checkpoint, 0),
            positive=get_value_at_index(positive, 0) if positive else None,
            negative=get_value_at_index(negative, 0) if negative else None,
            latent_image=get_value_at_index(vae_encoded, 0),
        )

        decoded = vaedecode.decode(
            samples=get_value_at_index(sampled, 0),
            vae=get_value_at_index(checkpoint, 2),
        )

        output_tensor = get_value_at_index(decoded, 0)
        output_image = tensor_to_pil(output_tensor)
        return output_image


def main():
    interface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="filepath", label="Input Image"),
        outputs=gr.Image(label="Output Image"),
        title="Image Transformation with ComfyUI",
        description="Upload an image to transform it using ComfyUI's pipeline.",
    )
    interface.launch(share=True)


if __name__ == "__main__":
    main()
