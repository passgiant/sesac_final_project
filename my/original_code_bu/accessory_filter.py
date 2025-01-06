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
    from comfy_utils.extra_config import load_extra_path_config

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
    '{"10": {"inputs": {"vae_name": "flux_vae.safetensors"}, "class_type": "VAELoader", "_meta": {"title": "Load VAE"}}, "11": {"inputs": {"clip_name1": "ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"}, "class_type": "DualCLIPLoader", "_meta": {"title": "DualCLIPLoader"}}, "12": {"inputs": {"unet_name": "flux1-fill-dev-FP8.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader", "_meta": {"title": "Load Diffusion Model"}}, "173": {"inputs": {"style_model_name": "flux1-redux-dev.safetensors"}, "class_type": "StyleModelLoader", "_meta": {"title": "Load Style Model"}}, "422": {"inputs": {"image": "christmas_headband_3.jpg", "upload": "image"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}}, "590": {"inputs": {"image": "586560_719472_347.jpg", "upload": "image"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}}, "641": {"inputs": {"padding": 80, "region_type": "dominant", "mask": ["590", 1]}, "class_type": "Mask Crop Region", "_meta": {"title": "Mask Crop Region"}}, "642": {"inputs": {"width": ["641", 6], "height": ["641", 7], "position": "top-left", "x_offset": ["641", 3], "y_offset": ["641", 2], "image": ["590", 0]}, "class_type": "ImageCrop+", "_meta": {"title": "\\ud83d\\udd27 Image Crop"}}, "643": {"inputs": {"images": ["642", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "644": {"inputs": {"mask": ["641", 0]}, "class_type": "MaskPreview+", "_meta": {"title": "\\ud83d\\udd27 Mask Preview"}}, "645": {"inputs": {"width": 768, "height": 0, "interpolation": "lanczos", "method": "keep proportion", "condition": "always", "multiple_of": 0, "image": ["642", 0]}, "class_type": "ImageResize+", "_meta": {"title": "\\ud83d\\udd27 Image Resize"}}, "646": {"inputs": {"width": ["645", 1], "height": ["645", 2], "interpolation": "lanczos", "method": "fill / crop", "condition": "always", "multiple_of": 0, "image": ["422", 0]}, "class_type": "ImageResize+", "_meta": {"title": "\\ud83d\\udd27 Image Resize"}}, "647": {"inputs": {"images": ["646", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "648": {"inputs": {"images": ["645", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "649": {"inputs": {"direction": "right", "match_image_size": true, "image1": ["646", 0], "image2": ["645", 0]}, "class_type": "ImageConcanate", "_meta": {"title": "Image Concatenate"}}, "650": {"inputs": {"images": ["649", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "651": {"inputs": {"width": ["645", 1], "height": ["645", 2], "red": 0, "green": 0, "blue": 0}, "class_type": "Image Blank", "_meta": {"title": "Image Blank"}}, "652": {"inputs": {"direction": "right", "match_image_size": true, "image1": ["651", 0], "image2": ["653", 0]}, "class_type": "ImageConcanate", "_meta": {"title": "Image Concatenate"}}, "653": {"inputs": {"mask": ["641", 0]}, "class_type": "MaskToImage", "_meta": {"title": "Convert Mask to Image"}}, "655": {"inputs": {"channel": "red", "image": ["652", 0]}, "class_type": "ImageToMask", "_meta": {"title": "Convert Image to Mask"}}, "656": {"inputs": {"mask": ["679", 0]}, "class_type": "MaskPreview+", "_meta": {"title": "\\ud83d\\udd27 Mask Preview"}}, "658": {"inputs": {"text": "", "clip": ["11", 0]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Prompt)"}}, "659": {"inputs": {"conditioning": ["658", 0]}, "class_type": "ConditioningZeroOut", "_meta": {"title": "ConditioningZeroOut"}}, "660": {"inputs": {"noise_mask": true, "positive": ["658", 0], "negative": ["659", 0], "vae": ["10", 0], "pixels": ["649", 0], "mask": ["679", 0]}, "class_type": "InpaintModelConditioning", "_meta": {"title": "InpaintModelConditioning"}}, "661": {"inputs": {"guidance": 50, "conditioning": ["667", 0]}, "class_type": "FluxGuidance", "_meta": {"title": "FluxGuidance"}}, "662": {"inputs": {"crop": "center", "clip_vision": ["663", 0], "image": ["422", 0]}, "class_type": "CLIPVisionEncode", "_meta": {"title": "CLIP Vision Encode"}}, "663": {"inputs": {"clip_name": "sigclip_vision_patch14_384.safetensors"}, "class_type": "CLIPVisionLoader", "_meta": {"title": "Load CLIP Vision"}}, "664": {"inputs": {"seed": ["685", 0], "steps": 20, "cfg": 1, "sampler_name": "euler", "scheduler": "normal", "denoise": 1, "model": ["12", 0], "positive": ["661", 0], "negative": ["660", 1], "latent_image": ["660", 2]}, "class_type": "KSampler", "_meta": {"title": "KSampler"}}, "665": {"inputs": {"samples": ["664", 0], "vae": ["10", 0]}, "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"}}, "666": {"inputs": {"images": ["665", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "667": {"inputs": {"strength": 1, "strength_type": "multiply", "conditioning": ["660", 0], "style_model": ["173", 0], "clip_vision_output": ["662", 0]}, "class_type": "StyleModelApply", "_meta": {"title": "Apply Style Model"}}, "669": {"inputs": {"images": ["680", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "676": {"inputs": {"x": ["641", 3], "y": ["641", 2], "resize_source": false, "destination": ["590", 0], "source": ["682", 0], "mask": ["683", 0]}, "class_type": "ImageCompositeMasked", "_meta": {"title": "ImageCompositeMasked"}}, "677": {"inputs": {"images": ["676", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Preview Image"}}, "679": {"inputs": {"expand": 6, "incremental_expandrate": 0, "tapered_corners": true, "flip_input": false, "blur_radius": 3, "lerp_alpha": 1, "decay_factor": 1, "fill_holes": false, "mask": ["655", 0]}, "class_type": "GrowMaskWithBlur", "_meta": {"title": "Grow Mask With Blur"}}, "680": {"inputs": {"width": ["645", 1], "height": ["645", 2], "position": "top-right", "x_offset": 0, "y_offset": 0, "image": ["665", 0]}, "class_type": "ImageCrop+", "_meta": {"title": "\\ud83d\\udd27 Image Crop"}}, "682": {"inputs": {"width": ["641", 6], "height": ["641", 7], "interpolation": "lanczos", "method": "keep proportion", "condition": "always", "multiple_of": 0, "image": ["680", 0]}, "class_type": "ImageResize+", "_meta": {"title": "\\ud83d\\udd27 Image Resize"}}, "683": {"inputs": {"expand": 3, "incremental_expandrate": 0, "tapered_corners": true, "flip_input": false, "blur_radius": 3, "lerp_alpha": 1, "decay_factor": 1, "fill_holes": false, "mask": ["641", 0]}, "class_type": "GrowMaskWithBlur", "_meta": {"title": "Grow Mask With Blur"}}, "684": {"inputs": {"rgthree_comparer": {"images": [{"name": "A", "selected": true, "url": "/api/view?filename=rgthree.compare._temp_ulhkj_00001_.png&type=temp&subfolder=&rand=0.8620886883487457"}, {"name": "B", "selected": true, "url": "/api/view?filename=rgthree.compare._temp_ulhkj_00002_.png&type=temp&subfolder=&rand=0.8534844584765022"}]}, "image_a": ["590", 0], "image_b": ["676", 0]}, "class_type": "Image Comparer (rgthree)", "_meta": {"title": "Image Comparer (rgthree)"}}, "685": {"inputs": {"seed": 989107267427890}, "class_type": "Seed (rgthree)", "_meta": {"title": "Seed (rgthree)"}}, "687": {"inputs": {"filename_prefix": "result_cloth1", "images": ["676", 0]}, "class_type": "SaveImage", "_meta": {"title": "Save Image"}}}'
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
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_10 = vaeloader.load_vae(vae_name="flux_vae.safetensors")

        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_11 = dualcliploader.load_clip(
            clip_name1="ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors",
            clip_name2="t5xxl_fp8_e4m3fn.safetensors",
            type="flux",
        )

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_12 = unetloader.load_unet(
            unet_name="flux1-fill-dev-FP8.safetensors", weight_dtype="default"
        )

        stylemodelloader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()
        stylemodelloader_173 = stylemodelloader.load_style_model(
            style_model_name="flux1-redux-dev.safetensors"
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_422 = loadimage.load_image(image="christmas_headband_3.jpg")

        loadimage_590 = loadimage.load_image(image="아이유_colored.jpg") # 586560_719472_347
        
        # print("loadimage_590:", loadimage_590)

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_658 = cliptextencode.encode(
            text="", clip=get_value_at_index(dualcliploader_11, 0)
        )

        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        conditioningzeroout_659 = conditioningzeroout.zero_out(
            conditioning=get_value_at_index(cliptextencode_658, 0)
        )

        mask_crop_region = NODE_CLASS_MAPPINGS["Mask Crop Region"]()
        mask_crop_region_641 = mask_crop_region.mask_crop_region(
            padding=80,
            region_type="dominant",
            mask=get_value_at_index(loadimage_590, 1),
        )

        imagecrop = NODE_CLASS_MAPPINGS["ImageCrop+"]()
        imagecrop_642 = imagecrop.execute(
            width=get_value_at_index(mask_crop_region_641, 6),
            height=get_value_at_index(mask_crop_region_641, 7),
            position="top-left",
            x_offset=get_value_at_index(mask_crop_region_641, 3),
            y_offset=get_value_at_index(mask_crop_region_641, 2),
            image=get_value_at_index(loadimage_590, 0),
        )

        imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
        imageresize_645 = imageresize.execute(
            width=768,
            height=0,
            interpolation="lanczos",
            method="keep proportion",
            condition="always",
            multiple_of=0,
            image=get_value_at_index(imagecrop_642, 0),
        )

        imageresize_646 = imageresize.execute(
            width=get_value_at_index(imageresize_645, 1),
            height=get_value_at_index(imageresize_645, 2),
            interpolation="lanczos",
            method="fill / crop",
            condition="always",
            multiple_of=0,
            image=get_value_at_index(loadimage_422, 0),
        )

        imageconcanate = NODE_CLASS_MAPPINGS["ImageConcanate"]()
        imageconcanate_649 = imageconcanate.concanate(
            direction="right",
            match_image_size=True,
            image1=get_value_at_index(imageresize_646, 0),
            image2=get_value_at_index(imageresize_645, 0),
        )

        image_blank = NODE_CLASS_MAPPINGS["Image Blank"]()
        image_blank_651 = image_blank.blank_image(
            width=get_value_at_index(imageresize_645, 1),
            height=get_value_at_index(imageresize_645, 2),
            red=0,
            green=0,
            blue=0,
        )

        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        masktoimage_653 = masktoimage.mask_to_image(
            mask=get_value_at_index(mask_crop_region_641, 0)
        )

        imageconcanate_652 = imageconcanate.concanate(
            direction="right",
            match_image_size=True,
            image1=get_value_at_index(image_blank_651, 0),
            image2=get_value_at_index(masktoimage_653, 0),
        )

        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        imagetomask_655 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(imageconcanate_652, 0)
        )

        growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
        growmaskwithblur_679 = growmaskwithblur.expand_mask(
            expand=6,
            incremental_expandrate=0,
            tapered_corners=True,
            flip_input=False,
            blur_radius=3,
            lerp_alpha=1,
            decay_factor=1,
            fill_holes=False,
            mask=get_value_at_index(imagetomask_655, 0),
        )

        inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
        inpaintmodelconditioning_660 = inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(cliptextencode_658, 0),
            negative=get_value_at_index(conditioningzeroout_659, 0),
            vae=get_value_at_index(vaeloader_10, 0),
            pixels=get_value_at_index(imageconcanate_649, 0),
            mask=get_value_at_index(growmaskwithblur_679, 0),
        )

        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        clipvisionloader_663 = clipvisionloader.load_clip(
            clip_name="sigclip_vision_patch14_384.safetensors"
        )

        clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
        clipvisionencode_662 = clipvisionencode.encode(
            crop="center",
            clip_vision=get_value_at_index(clipvisionloader_663, 0),
            image=get_value_at_index(loadimage_422, 0),
        )

        seed_rgthree = NODE_CLASS_MAPPINGS["Seed (rgthree)"]()
        seed_rgthree_685 = seed_rgthree.main(
            seed=random.randint(1, 2**64),
            unique_id=1157898235765167819,
            prompt=PROMPT_DATA,
        )

        maskpreview = NODE_CLASS_MAPPINGS["MaskPreview+"]()
        stylemodelapply = NODE_CLASS_MAPPINGS["StyleModelApply"]()
        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        image_comparer_rgthree = NODE_CLASS_MAPPINGS["Image Comparer (rgthree)"]()
        saveimage = save_image_wrapper(ctx, NODE_CLASS_MAPPINGS["SaveImage"])()
        for q in range(args.queue_size):
            maskpreview_644 = maskpreview.execute(
                mask=get_value_at_index(mask_crop_region_641, 0), prompt=PROMPT_DATA
            )

            maskpreview_656 = maskpreview.execute(
                mask=get_value_at_index(growmaskwithblur_679, 0), prompt=PROMPT_DATA
            )

            stylemodelapply_667 = stylemodelapply.apply_stylemodel(
                strength=1,
                strength_type="multiply",
                conditioning=get_value_at_index(inpaintmodelconditioning_660, 0),
                style_model=get_value_at_index(stylemodelloader_173, 0),
                clip_vision_output=get_value_at_index(clipvisionencode_662, 0),
            )

            fluxguidance_661 = fluxguidance.append(
                guidance=50, conditioning=get_value_at_index(stylemodelapply_667, 0)
            )

            ksampler_664 = ksampler.sample(
                seed=get_value_at_index(seed_rgthree_685, 0),
                steps=20,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(unetloader_12, 0),
                positive=get_value_at_index(fluxguidance_661, 0),
                negative=get_value_at_index(inpaintmodelconditioning_660, 1),
                latent_image=get_value_at_index(inpaintmodelconditioning_660, 2),
            )

            vaedecode_665 = vaedecode.decode(
                samples=get_value_at_index(ksampler_664, 0),
                vae=get_value_at_index(vaeloader_10, 0),
            )

            imagecrop_680 = imagecrop.execute(
                width=get_value_at_index(imageresize_645, 1),
                height=get_value_at_index(imageresize_645, 2),
                position="top-right",
                x_offset=0,
                y_offset=0,
                image=get_value_at_index(vaedecode_665, 0),
            )

            imageresize_682 = imageresize.execute(
                width=get_value_at_index(mask_crop_region_641, 6),
                height=get_value_at_index(mask_crop_region_641, 7),
                interpolation="lanczos",
                method="keep proportion",
                condition="always",
                multiple_of=0,
                image=get_value_at_index(imagecrop_680, 0),
            )

            growmaskwithblur_683 = growmaskwithblur.expand_mask(
                expand=3,
                incremental_expandrate=0,
                tapered_corners=True,
                flip_input=False,
                blur_radius=3,
                lerp_alpha=1,
                decay_factor=1,
                fill_holes=False,
                mask=get_value_at_index(mask_crop_region_641, 0),
            )

            imagecompositemasked_676 = imagecompositemasked.composite(
                x=get_value_at_index(mask_crop_region_641, 3),
                y=get_value_at_index(mask_crop_region_641, 2),
                resize_source=False,
                destination=get_value_at_index(loadimage_590, 0),
                source=get_value_at_index(imageresize_682, 0),
                mask=get_value_at_index(growmaskwithblur_683, 0),
            )

            image_comparer_rgthree_684 = image_comparer_rgthree.compare_images(
                image_a=get_value_at_index(loadimage_590, 0),
                image_b=get_value_at_index(imagecompositemasked_676, 0),
                prompt=PROMPT_DATA,
            )

            if __name__ != "__main__":
                return dict(
                    filename_prefix="result_cloth1",
                    images=get_value_at_index(imagecompositemasked_676, 0),
                    prompt=PROMPT_DATA,
                )
            else:
                saveimage_687 = saveimage.save_images(
                    filename_prefix="result_cloth1",
                    images=get_value_at_index(imagecompositemasked_676, 0),
                    prompt=PROMPT_DATA,
                )


if __name__ == "__main__":
    main()
