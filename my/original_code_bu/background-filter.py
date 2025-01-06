import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import torch


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
    # If no path is given, use the current working directory
    if path is None:
        if args is None or args.comfyui_directory is None:
            path = os.getcwd()
        else:
            path = args.comfyui_directory

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes(init_custom_nodes=True)


def save_image_wrapper(context, cls):
    if args.output is None:
        return cls

    from PIL import Image, ImageOps, ImageSequence
    from PIL.PngImagePlugin import PngInfo

    import numpy as np

    class WrappedSaveImage(cls):
        counter = 0

        def save_images(
            self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
        ):
            if args.output is None:
                return super().save_images(
                    images, filename_prefix, prompt, extra_pnginfo
                )
            else:
                if len(images) > 1 and args.output == "-":
                    raise ValueError("Cannot save multiple images to stdout")
                filename_prefix += self.prefix_append

                results = list()
                for batch_number, image in enumerate(images):
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    metadata = None
                    if not args.disable_metadata:
                        metadata = PngInfo()
                        if prompt is not None:
                            metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                    if args.output == "-":
                        # Hack to briefly restore stdout
                        if context is not None:
                            context.__exit__(None, None, None)
                        try:
                            img.save(
                                sys.stdout.buffer,
                                format="png",
                                pnginfo=metadata,
                                compress_level=self.compress_level,
                            )
                        finally:
                            if context is not None:
                                context.__enter__()
                    else:
                        subfolder = ""
                        if len(images) == 1:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = "output.png"
                            else:
                                subfolder, file = os.path.split(args.output)
                                if subfolder == "":
                                    subfolder = os.getcwd()
                        else:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = filename_prefix
                            else:
                                subfolder, file = os.path.split(args.output)

                            if subfolder == "":
                                subfolder = os.getcwd()

                            files = os.listdir(subfolder)
                            file_pattern = file
                            while True:
                                filename_with_batch_num = file_pattern.replace(
                                    "%batch_num%", str(batch_number)
                                )
                                file = (
                                    f"{filename_with_batch_num}_{self.counter:05}.png"
                                )
                                self.counter += 1

                                if file not in files:
                                    break

                        img.save(
                            os.path.join(subfolder, file),
                            pnginfo=metadata,
                            compress_level=self.compress_level,
                        )
                        print("Saved image to", os.path.join(subfolder, file))
                        results.append(
                            {
                                "filename": file,
                                "subfolder": subfolder,
                                "type": self.type,
                            }
                        )

                return {"ui": {"images": results}}

    return WrappedSaveImage


def parse_arg(s: Any):
    """Parses a JSON string, returning it unchanged if the parsing fails."""
    if __name__ == "__main__" or not isinstance(s, str):
        return s

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


parser = argparse.ArgumentParser(
    description="A converted ComfyUI workflow. Required inputs listed below. Values passed should be valid JSON (assumes string if not valid JSON)."
)
parser.add_argument(
    "--queue-size",
    "-q",
    type=int,
    default=1,
    help="How many times the workflow will be executed (default: 1)",
)

parser.add_argument(
    "--comfyui-directory",
    "-c",
    default=None,
    help="Where to look for ComfyUI (default: current directory)",
)

parser.add_argument(
    "--output",
    "-o",
    default=None,
    help="The location to save the output image. Either a file path, a directory, or - for stdout (default: the ComfyUI output directory)",
)

parser.add_argument(
    "--disable-metadata",
    action="store_true",
    help="Disables writing workflow metadata to the outputs",
)


comfy_args = [sys.argv[0]]
if __name__ == "__main__" and "--" in sys.argv:
    idx = sys.argv.index("--")
    comfy_args += sys.argv[idx + 1 :]
    sys.argv = sys.argv[:idx]

args = None
if __name__ == "__main__":
    args = parser.parse_args()
    sys.argv = comfy_args
if args is not None and args.output is not None and args.output == "-":
    ctx = contextlib.redirect_stdout(sys.stderr)
else:
    ctx = contextlib.nullcontext()

PROMPT_DATA = json.loads(
    '{"1": {"inputs": {"image": "586560_719472_347.jpg", "upload": "image"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}}, "3": {"inputs": {"torchscript_jit": "default", "image": ["1", 0]}, "class_type": "InspyrenetRembg", "_meta": {"title": "Inspyrenet Rembg"}}, "4": {"inputs": {"images": ["3", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "14": {"inputs": {"mask": ["3", 1]}, "class_type": "MaskToImage", "_meta": {"title": "Convert Mask to Image"}}, "15": {"inputs": {"images": ["14", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "16": {"inputs": {"image": ["3", 0]}, "class_type": "SplitImageWithAlpha", "_meta": {"title": "Split Image with Alpha"}}, "17": {"inputs": {"grow_mask_by": 9, "pixels": ["1", 0], "vae": ["18", 2], "mask": ["16", 1]}, "class_type": "VAEEncodeForInpaint", "_meta": {"title": "VAE Encode (for Inpainting)"}}, "18": {"inputs": {"ckpt_name": "realisticVisionV51_v51VAE.safetensors"}, "class_type": "CheckpointLoaderSimple", "_meta": {"title": "Load Checkpoint"}}, "19": {"inputs": {"text": "A festive Christma sparty scene with people wearing cozy sweaters, holding hot cocoa, and exchanging gifts under colorful Christmas lights. A Christmas tree adorned with golden ribbons and ornaments.", "clip": ["18", 1]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Prompt)"}}, "20": {"inputs": {"text": "distortion, ugly, animation, 4K, blurry image, low resolution, artifacts, overexposure, grainy", "clip": ["18", 1]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Prompt)"}}, "21": {"inputs": {"seed": 776455704192158, "steps": 40, "cfg": 8, "sampler_name": "euler", "scheduler": "normal", "denoise": 0.55, "model": ["18", 0], "positive": ["19", 0], "negative": ["20", 0], "latent_image": ["17", 0]}, "class_type": "KSampler", "_meta": {"title": "KSampler"}}, "22": {"inputs": {"samples": ["21", 0], "vae": ["18", 2]}, "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"}}, "24": {"inputs": {"images": ["22", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "25": {"inputs": {"filename_prefix": "result_background1", "images": ["22", 0]}, "class_type": "SaveImage", "_meta": {"title": "Save Image"}}}'
)


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes(init_custom_nodes=True)


_custom_nodes_imported = False
_custom_path_added = False


def main(*func_args, **func_kwargs):
    global args, _custom_nodes_imported, _custom_path_added
    if __name__ == "__main__":
        if args is None:
            args = parser.parse_args()
    else:
        defaults = dict(
            (arg, parser.get_default(arg))
            for arg in ["queue_size", "comfyui_directory", "output", "disable_metadata"]
        )
        ordered_args = dict(zip([], func_args))

        all_args = dict()
        all_args.update(defaults)
        all_args.update(ordered_args)
        all_args.update(func_kwargs)

        args = argparse.Namespace(**all_args)

    with ctx:
        if not _custom_path_added:
            add_comfyui_directory_to_sys_path()
            add_extra_model_paths()

            _custom_path_added = True

        if not _custom_nodes_imported:
            import_custom_nodes()

            _custom_nodes_imported = True

        from nodes import NODE_CLASS_MAPPINGS

    with torch.inference_mode(), ctx:
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_1 = loadimage.load_image(image="586560_719472_347.jpg")

        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_18 = checkpointloadersimple.load_checkpoint(
            ckpt_name="realisticVisionV51_v51VAE.safetensors"
        )

        inspyrenetrembg = NODE_CLASS_MAPPINGS["InspyrenetRembg"]()
        inspyrenetrembg_3 = inspyrenetrembg.remove_background(
            torchscript_jit="default", image=get_value_at_index(loadimage_1, 0)
        )

        splitimagewithalpha = NODE_CLASS_MAPPINGS["SplitImageWithAlpha"]()
        splitimagewithalpha_16 = splitimagewithalpha.split_image_with_alpha(
            image=get_value_at_index(inspyrenetrembg_3, 0)
        )

        vaeencodeforinpaint = NODE_CLASS_MAPPINGS["VAEEncodeForInpaint"]()
        vaeencodeforinpaint_17 = vaeencodeforinpaint.encode(
            grow_mask_by=9,
            pixels=get_value_at_index(loadimage_1, 0),
            vae=get_value_at_index(checkpointloadersimple_18, 2),
            mask=get_value_at_index(splitimagewithalpha_16, 1),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_19 = cliptextencode.encode(
            text="A festive Christma sparty scene with people wearing cozy sweaters, holding hot cocoa, and exchanging gifts under colorful Christmas lights. A Christmas tree adorned with golden ribbons and ornaments.",
            clip=get_value_at_index(checkpointloadersimple_18, 1),
        )

        cliptextencode_20 = cliptextencode.encode(
            text="distortion, ugly, animation, 4K, blurry image, low resolution, artifacts, overexposure, grainy",
            clip=get_value_at_index(checkpointloadersimple_18, 1),
        )

        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = save_image_wrapper(ctx, NODE_CLASS_MAPPINGS["SaveImage"])()
        for q in range(args.queue_size):
            masktoimage_14 = masktoimage.mask_to_image(
                mask=get_value_at_index(inspyrenetrembg_3, 1)
            )

            ksampler_21 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=40,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.55,
                model=get_value_at_index(checkpointloadersimple_18, 0),
                positive=get_value_at_index(cliptextencode_19, 0),
                negative=get_value_at_index(cliptextencode_20, 0),
                latent_image=get_value_at_index(vaeencodeforinpaint_17, 0),
            )

            vaedecode_22 = vaedecode.decode(
                samples=get_value_at_index(ksampler_21, 0),
                vae=get_value_at_index(checkpointloadersimple_18, 2),
            )

            if __name__ != "__main__":
                return dict(
                    filename_prefix="result_background1",
                    images=get_value_at_index(vaedecode_22, 0),
                    prompt=PROMPT_DATA,
                )
            else:
                saveimage_25 = saveimage.save_images(
                    filename_prefix="result_background1",
                    images=get_value_at_index(vaedecode_22, 0),
                    prompt=PROMPT_DATA,
                )


if __name__ == "__main__":
    main()
