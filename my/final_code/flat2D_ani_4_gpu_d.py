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
        if comfyui_path not in sys.path:
            sys.path.append(comfyui_path)
            print(f"'{comfyui_path}' added to sys.path")
        else:
            print(f"'{comfyui_path}' already in sys.path")

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

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    array = tensor.cpu().numpy()
    array = (array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)

def process_image(image_array: np.ndarray):
    import_custom_nodes()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from nodes import NODE_CLASS_MAPPINGS

    def transform(image_pil, seed):
        try:
            with torch.inference_mode():
                # LoadImage
                loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
                loadimage_result = loadimage.load_image(image=image_pil)

                # Miaoshouai_Tagger
                tagger = NODE_CLASS_MAPPINGS["Miaoshouai_Tagger"]()
                tagger_result = tagger.start_tag(
                    model="promptgen_base_v1.5",
                    folder_path="Path to your image folder",
                    caption_method="detailed",
                    max_new_tokens=1024,
                    num_beams=4,
                    random_prompt="never",
                    prefix_caption="",
                    suffix_caption="",
                    replace_tags="",
                    images=get_value_at_index(loadimage_result, 0),
                )

                tags = get_value_at_index(tagger_result, 2)
                print(f"Generated Tags: {tags}")

                # CheckpointLoader
                checkpoint_loader = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
                checkpoint = checkpoint_loader.load_checkpoint(ckpt_name="flat2DAnimerge_v45Sharp.safetensors")

                # CLIPSetLastLayer
                clipsetlastlayer = NODE_CLASS_MAPPINGS["CLIPSetLastLayer"]()
                clipsetlastlayer_result = clipsetlastlayer.set_last_layer(
                    stop_at_clip_layer=-2, clip=get_value_at_index(checkpoint, 1)
                )

                # CLIPTextEncode
                clip_encoder = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
                positive = clip_encoder.encode(
                    text=tags, clip=get_value_at_index(clipsetlastlayer_result, 0)
                )
                negative = clip_encoder.encode(
                    text="verybadimagenegative_v1.3", clip=get_value_at_index(checkpoint, 1)
                )

                # VAEEncode
                vae_encoder = NODE_CLASS_MAPPINGS["VAEEncode"]()
                vae_encoded = vae_encoder.encode(
                    pixels=get_value_at_index(loadimage_result, 0).to(device),
                    vae=get_value_at_index(checkpoint, 2)
                )

                # KSampler
                ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
                sampled = ksampler.sample(
                    seed=seed,
                    steps=30,
                    cfg=7,
                    sampler_name="dpmpp_2m_sde_gpu",
                    scheduler="karras",
                    denoise=0.45,
                    model=get_value_at_index(checkpoint, 0),
                    positive=get_value_at_index(positive, 0),
                    negative=get_value_at_index(negative, 0),
                    latent_image=get_value_at_index(vae_encoded, 0),
                )

                # VAEDecode
                vae_decoder = NODE_CLASS_MAPPINGS["VAEDecode"]()
                decoded = vae_decoder.decode(
                    samples=get_value_at_index(sampled, 0),
                    vae=get_value_at_index(checkpoint, 2)
                )

                output_tensor = get_value_at_index(decoded, 0)
                return tensor_to_pil(output_tensor)

        except Exception as e:
            print(f"Error in transform: {e}")
            return None

    image_pil = Image.fromarray(image_array.astype(np.uint8))
    output_images = [
        transform(image_pil, seed=random.randint(1, 2**64)) for _ in range(4)
    ]
    return output_images

def gradio_interface(image_path):
    return process_image(image_path)

def main():
    interface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Image(type="filepath", label="Input Image"),
        outputs=[
            gr.Image(label="Output Image 1"),
            gr.Image(label="Output Image 2"),
            gr.Image(label="Output Image 3"),
            gr.Image(label="Output Image 4")
        ],
        title="Image Transformation Pipeline with CLIPSetLastLayer",
        description="Upload an image to generate tags and transformed outputs using Miaoshouai_Tagger, CLIPSetLastLayer, and ComfyUI pipelines."
    )
    interface.launch(share=True)

if __name__ == "__main__":
    add_comfyui_directory_to_sys_path()
    main()
