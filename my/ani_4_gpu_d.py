import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import numpy as np
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
        if comfyui_path not in sys.path:  # 중복 추가 방지
            sys.path.append(comfyui_path)
            print(f"'{comfyui_path}' added to sys.path")
        else:
            print(f"'{comfyui_path}' already in sys.path")

add_comfyui_directory_to_sys_path()

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

def tensor_to_pil(tensor):
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    array = tensor.cpu().numpy()
    array = (array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)

from nodes import NODE_CLASS_MAPPINGS

def process_image_from_array(image_array: np.ndarray):
    """
    이미지 배열을 입력으로 받아 모델을 통해 4개의 결과 이미지를 반환.
    Args:
        image_array (np.ndarray): 입력 이미지 (H, W, C).

    Returns:
        List[Image]: 변환된 이미지 4장
    """
    import_custom_nodes()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def transform(image, seed):
        try:
            with torch.inference_mode():
                loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
                image_pil = Image.fromarray(image_array.astype(np.uint8))
                loadimage_10 = loadimage.load_image(image=image_pil)

                checkpointloader = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
                checkpoint = checkpointloader.load_checkpoint(
                    ckpt_name="catCitronAnimeTreasure_v10.safetensors"
                )

                vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
                vae_encoded = vaeencode.encode(
                    pixels=get_value_at_index(loadimage_10, 0).to(device),
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
                sampled = ksampler.sample(
                    seed=seed,
                    steps=30,
                    cfg=7,
                    sampler_name="euler_ancestral",
                    scheduler="karras",
                    denoise=0.45,
                    model=get_value_at_index(checkpoint, 0),
                    positive=get_value_at_index(positive, 0),
                    negative=get_value_at_index(negative, 0),
                    latent_image=get_value_at_index(vae_encoded, 0),
                )

                if isinstance(sampled, tuple) and isinstance(sampled[0], dict):
                    samples = sampled[0]
                else:
                    raise ValueError("Unexpected structure for sampled output.")

                vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
                decoded = vaedecode.decode(
                    vae=get_value_at_index(checkpoint, 2),
                    samples=samples
                )

                output_tensor = get_value_at_index(decoded, 0)
                return tensor_to_pil(output_tensor)

        except Exception as e:
            print(f"Error in transform function: {e}")
            raise

    output_images = [
        transform(image_array, seed=random.randint(1, 2**64)) for _ in range(4)
    ]
    return output_images

def main():
    def gradio_interface(image):
        return process_image_from_array(np.array(image))

    interface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Image(type="pil", label="Input Image"),
        outputs=[
            gr.Image(label="Output Image 1"),
            gr.Image(label="Output Image 2"),
            gr.Image(label="Output Image 3"),
            gr.Image(label="Output Image 4")
        ],
        title="Image Transformation",
        description="Upload an image to process and generate four transformed outputs."
    )
    interface.launch(share=True)

if __name__ == "__main__":
    main()