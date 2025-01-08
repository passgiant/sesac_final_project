import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import numpy as np
import gradio as gr

# =========================
# Utility Functions
# =========================
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Retrieve value at index from sequence or mapping."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name: str, path: str = None) -> str:
    """Recursively look for a folder in parent directories."""
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
    """Add ComfyUI directory to sys.path."""
    comfyui_path = find_path("ComfyUI")
    if comfyui_path and os.path.isdir(comfyui_path):
        if comfyui_path not in sys.path:
            sys.path.append(comfyui_path)
            print(f"'{comfyui_path}' added to sys.path")

add_comfyui_directory_to_sys_path()

# =========================
# Node Imports and Setup
# =========================
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
    """Convert torch tensor to PIL Image."""
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    array = tensor.cpu().numpy()
    array = (array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)

from nodes import NODE_CLASS_MAPPINGS

# =========================
# Process Image
# =========================
def process_image_from_array(image_array: np.ndarray):
    """
    Process input image through the model pipeline.
    Args:
        image_array (np.ndarray): Input image as numpy array.
    Returns:
        List[PIL.Image]: List of four transformed images.
    """
    import_custom_nodes()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def transform(image, seed):
        with torch.inference_mode():
            # LoadImage Node
            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            loadimage_10 = loadimage.load_image(image=Image.fromarray(image))

            # Miaoshouai_Tagger Node
            miaoshouai_tagger = NODE_CLASS_MAPPINGS["Miaoshouai_Tagger"]()
            miaoshouai_tagger_18 = miaoshouai_tagger.start_tag(
                model="promptgen_base_v1.5",
                folder_path="Path to your image folder",
                caption_method="detailed",
                max_new_tokens=1024,
                num_beams=4,
                random_prompt="never",
                prefix_caption="",
                suffix_caption="",
                replace_tags="replace_tags eg:search1:replace1;search2:replace2",
                images=get_value_at_index(loadimage_10, 0),
            )

            tags = get_value_at_index(miaoshouai_tagger_18, 2)
            print(f"Generated Tags: {tags}")

            # CheckpointLoader Node
            checkpointloader = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            checkpoint = checkpointloader.load_checkpoint(
                ckpt_name="brmAnimeBeautyrealmix_v41.safetensors"
            )

            # CLIPSetLastLayer Node
            clipsetlastlayer = NODE_CLASS_MAPPINGS["CLIPSetLastLayer"]()
            clipsetlastlayer_result = clipsetlastlayer.set_last_layer(
                stop_at_clip_layer=-2, clip=get_value_at_index(checkpoint, 1)
            )

            # CLIPTextEncode Node
            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            positive = cliptextencode.encode(
                text=tags,
                clip=get_value_at_index(clipsetlastlayer_result, 0),
            )
            negative = cliptextencode.encode(
                text="verybadimagenegative_v1.3",
                clip=get_value_at_index(checkpoint, 1),
            )

            # VAEEncode Node
            vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
            vae_encoded = vaeencode.encode(
                pixels=get_value_at_index(loadimage_10, 0).to(device),
                vae=get_value_at_index(checkpoint, 2),
            )

            # KSampler Node
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

            # VAEDecode Node
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded = vaedecode.decode(
                vae=get_value_at_index(checkpoint, 2),
                samples=get_value_at_index(sampled, 0),
            )

            return tensor_to_pil(get_value_at_index(decoded, 0))

    output_images = [
        transform(image_array, seed=random.randint(1, 2**64)) for _ in range(4)
    ]
    return output_images

# =========================
# Gradio Interface
# =========================
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
        title="Image Transformation Pipeline with CLIPSetLastLayer",
        description="Upload an image to process and generate four transformed outputs using Miaoshouai_Tagger, CLIPSetLastLayer, and KSampler nodes.",
    )
    interface.launch(share=True)

if __name__ == "__main__":
    main()
