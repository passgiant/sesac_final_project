import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import numpy as np

# =========================
# 커스텀 노드 및 환경설정 로드
# =========================
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """
    Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
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
        if comfyui_path not in sys.path:
            sys.path.append(comfyui_path)
            print(f"'{comfyui_path}' added to sys.path")

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

# =========================
# 템서를 PIL 이미지로 변환
# =========================
def tensor_to_pil(tensor):
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    array = tensor.cpu().numpy()
    array = (array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)

from nodes import NODE_CLASS_MAPPINGS

# =========================
# 모델로 이미지 변환 함수
# =========================
def process_image_from_array(image_array: np.ndarray):
    """
    이미지 배열을 입력으로 받아 모델을 통해 4개의 결과 이미지를 반환.
    """
    import_custom_nodes()

    def transform(image, seed):
        with torch.inference_mode():
            # LoadImage 노드
            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            loadimage_10 = loadimage.load_image(image=Image.fromarray(image))

            # Miaoshouai_Tagger 노드
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

            # 텍스트 표시 노드
            showtext = NODE_CLASS_MAPPINGS["ShowText|pysssss"]()
            showtext_21 = showtext.notify(
                text=get_value_at_index(miaoshouai_tagger_18, 2),
                unique_id=7318135858042453857,
            )

            # CheckpointLoader 노드
            checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            checkpoint = checkpointloadersimple.load_checkpoint(
                ckpt_name="catCitronAnimeTreasure_v10.safetensors"
            )

            # CLIPSetLastLayer 노드 추가
            clipsetlastlayer = NODE_CLASS_MAPPINGS["CLIPSetLastLayer"]()
            clipsetlastlayer_result = clipsetlastlayer.set_last_layer(
                stop_at_clip_layer=-2, clip=get_value_at_index(checkpoint, 1)
            )

            # CLIPTextEncode 노드
            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            positive = cliptextencode.encode(
                text=get_value_at_index(showtext_21, 0),
                clip=get_value_at_index(clipsetlastlayer_result, 0),
            )
            negative = cliptextencode.encode(
                text="verybadimagenegative_v1.3",
                clip=get_value_at_index(checkpoint, 1),
            )

            # VAEEncode 노드
            vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
            vae_encoded = vaeencode.encode(
                pixels=get_value_at_index(loadimage_10, 0),
                vae=get_value_at_index(checkpoint, 2),
            )

            # KSampler 및 VAEDecode 노드
            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            sampled = ksampler.sample(
                seed=seed,
                steps=30,
                cfg=7,
                sampler_name="dpmpp_3m_sde_gpu",
                scheduler="karras",
                denoise=0.5,
                model=get_value_at_index(checkpoint, 0),
                positive=get_value_at_index(positive, 0),
                negative=get_value_at_index(negative, 0),
                latent_image=get_value_at_index(vae_encoded, 0),
            )

            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded = vaedecode.decode(
                vae=get_value_at_index(checkpoint, 2),
                samples=get_value_at_index(sampled, 0),
            )

            return tensor_to_pil(get_value_at_index(decoded, 0))

    # 4개의 결과 이미지 생성
    output_images = [transform(image_array, seed=random.randint(1, 2**64)) for _ in range(4)]
    return output_images

# =========================
# 테스트 코드
# =========================
if __name__ == "__main__":
    test_image_path = "아이유.jpg"
    input_image = Image.open(test_image_path).convert("RGB")
    input_image_array = np.array(input_image)

    output_images = process_image_from_array(input_image_array)

    for i, img in enumerate(output_images):
        img.save(f"output_image_{i+1}.png")
        print(f"Output Image {i+1} saved.")
