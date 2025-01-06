import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import numpy as np

# Utility to get value at an index
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

# Recursively find a path
def find_path(name: str, path: str = None) -> str:
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        return os.path.join(path, name)
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    return find_path(name, parent_directory)

# Add ComfyUI directory to sys.path
def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        if comfyui_path not in sys.path:
            sys.path.append(comfyui_path)
            print(f"'{comfyui_path}' added to sys.path")
add_comfyui_directory_to_sys_path()

# Import custom nodes
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

# Tensor to PIL converter
def tensor_to_pil(tensor):
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    array = tensor.cpu().numpy()
    array = (array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)

# Import NODE_CLASS_MAPPINGS
from nodes import NODE_CLASS_MAPPINGS

# Core processing function
def process_image_from_array(image_array: np.ndarray):
    import_custom_nodes()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.inference_mode():
        # Load image
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loaded_image = loadimage.load_image(image=Image.fromarray(image_array))

        # Generate tags
        miaoshouai_tagger = NODE_CLASS_MAPPINGS["Miaoshouai_Tagger"]()
        tags = miaoshouai_tagger.start_tag(
            model="promptgen_base_v1.5",
            folder_path="Path to your image folder",
            caption_method="detailed",
            max_new_tokens=1024,
            num_beams=4,
            random_prompt="never",
            prefix_caption="",
            suffix_caption="",
            replace_tags="replace_tags eg:search1:replace1;search2:replace2",
            images=get_value_at_index(loaded_image, 0),
        )
        tag_text = get_value_at_index(tags, 2)
        print("Tags Generated:", tag_text)

        # Load checkpoint
        checkpointloader = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpoint = checkpointloader.load_checkpoint(ckpt_name="realcartoonPixar_v12.safetensors")

        # Set CLIP last layer
        clipsetlastlayer = NODE_CLASS_MAPPINGS["CLIPSetLastLayer"]()
        clipsetlastlayer_15 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-2, clip=get_value_at_index(checkpoint, 1)
        )

        # VAE Encoding
        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        vae_encoded = vaeencode.encode(
            pixels=get_value_at_index(loaded_image, 0).to(device),
            vae=get_value_at_index(checkpoint, 2),
        )

        # CLIP Text Encoding
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        positive = cliptextencode.encode(
            text=tag_text, clip=get_value_at_index(clipsetlastlayer_15, 0)
        )
        negative = cliptextencode.encode(text="", clip=get_value_at_index(checkpoint, 1))

        # KSampler and VAE Decoding
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        images = []
        for _ in range(4):  # Generate 4 images
            sampled = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=30,
                cfg=7,
                sampler_name="dpmpp_3m_sde_gpu",
                scheduler="karras",
                denoise=0.45,
                model=get_value_at_index(checkpoint, 0),
                positive=get_value_at_index(positive, 0),
                negative=get_value_at_index(negative, 0),
                latent_image=get_value_at_index(vae_encoded, 0),
            )

            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded_image = vaedecode.decode(
                samples=get_value_at_index(sampled, 0), vae=get_value_at_index(checkpoint, 2)
            )
            images.append(tensor_to_pil(get_value_at_index(decoded_image, 0)))

        return images  # Return a list of 4 images

# Main execution
if __name__ == "__main__":
    input_image_path = "input_image.jpg"
    input_image = Image.open(input_image_path).convert("RGB")
    input_array = np.array(input_image)

    output_images = process_image_from_array(input_array)

    # Save and display results
    for i, img in enumerate(output_images):
        output_path = f"output_image_{i+1}.png"
        img.save(output_path)
        print(f"Saved: {output_path}")
