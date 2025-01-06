import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import gradio as gr


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
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

import numpy as np

def tensor_to_numpy(tensor):
    """
    Converts a PyTorch tensor to a NumPy array suitable for PIL.
    Args:
        tensor (torch.Tensor): The tensor to convert. Expected shape: (B, H, W, C) or (H, W, C).

    Returns:
        np.ndarray: Converted NumPy array with dtype uint8 and proper shape.
    """
    # Convert to NumPy array
    array = tensor.cpu().numpy()

    # Remove batch dimension if present (B, H, W, C -> H, W, C)
    if len(array.shape) == 4 and array.shape[0] == 1:
        array = array[0]  # Remove the batch dimension

    # Validate shape (should be H, W, C)
    if len(array.shape) != 3 or array.shape[-1] != 3:
        raise ValueError(f"Invalid shape for RGB conversion: {array.shape}")

    # Scale to [0, 255] and convert to uint8
    array = (array * 255).clip(0, 255).astype(np.uint8)

    return array




def process_image(image_path: str):
    try:
        import_custom_nodes()
        with torch.inference_mode():
            # Load Image
            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            loadimage_10 = loadimage.load_image(image=image_path)
            if not loadimage_10:
                return np.zeros((256, 256, 3), dtype=np.uint8)

            # Load Checkpoint
            checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            checkpointloadersimple_14 = checkpointloadersimple.load_checkpoint(
                ckpt_name="catCitronAnimeTreasure_v10.safetensors"
            )
            if not checkpointloadersimple_14:
                return np.zeros((256, 256, 3), dtype=np.uint8)

            # Encode Text (Positive)
            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            positive = cliptextencode.encode(
                text="Generated description",
                clip=get_value_at_index(checkpointloadersimple_14, 1),
            ) or []
            if not positive:
                positive = []

            # Encode Text (Negative)
            negative = cliptextencode.encode(
                text="Negative description",
                clip=get_value_at_index(checkpointloadersimple_14, 1),
            ) or []
            if not negative:
                negative = []

            # VAE Encoding
            vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
            vaeencode_12 = vaeencode.encode(
                pixels=get_value_at_index(loadimage_10, 0),
                vae=get_value_at_index(checkpointloadersimple_14, 2),
            )
            if not vaeencode_12:
                return np.zeros((256, 256, 3), dtype=np.uint8)

            # Sampling
            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=30,
                cfg=7,
                sampler_name="dpmpp_3m_sde_gpu",
                scheduler="karras",
                denoise=0.5,
                model=get_value_at_index(checkpointloadersimple_14, 0),
                positive=get_value_at_index(positive, 0) if positive else [],
                negative=get_value_at_index(negative, 0) if negative else [],
                latent_image=get_value_at_index(vaeencode_12, 0),
            )
            if not ksampler_3:
                return np.zeros((256, 256, 3), dtype=np.uint8)

            # VAE Decoding
            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_14, 2),
            )
            if not vaedecode_8:
                return np.zeros((256, 256, 3), dtype=np.uint8)

            # Convert tensor to a NumPy array
            output_tensor = get_value_at_index(vaedecode_8, 0)
            output_array = tensor_to_numpy(output_tensor)

            return output_array
    except Exception as e:
        return np.zeros((256, 256, 3), dtype=np.uint8)  # Default black image


# Gradio Interface
import gradio as gr

def main():
    interface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="filepath", label="Input Image"),
        outputs=gr.Image(label="Output Image"),
        title="Image Transformation",
        description="Upload an image to process and transform it.",
    )
    interface.launch(share=True)



if __name__ == "__main__":
    main()
