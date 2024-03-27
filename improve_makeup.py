import argparse
from pathlib import Path

from PIL import Image
from psgan import Inference
from fire import Fire
import numpy as np

import faceutils as futils
from psgan import PostProcess
from setup import setup_config, setup_argparser

import torch

import logging


def improve_makeup(no_makeup_path, makeup_path, result_path, do_rewrite=True,
                   device="cpu", model_path="assets/models/G.pth"):
    """
    Applies PSGAN to transfer makeup from one photo to another.
    Works only with squared pictures.
    Stores the result in result_path.
    Returns True if the application was successful, else returns False
    """
    no_makeup_file = Path(no_makeup_path)
    makeup_file = Path(makeup_path)
    result_file = Path(result_path)

    if not no_makeup_file.is_file():
        logging.info(f"File {no_makeup_path} does not exist")
        return False

    if not makeup_file.is_file():
        logging.info(f"File {makeup_path} does not exist")
        return False

    if result_file.is_file():
        if not do_rewrite:
            logging.info(f"File {result_path} exists, but do_rewrite=False")
            return False

    parser = setup_argparser()
    args = parser.parse_args()
    config = setup_config(args)

    inference = Inference(config, device, model_path)
    postprocess = PostProcess(config)

    no_makeup_img = Image.open(no_makeup_path).convert("RGB")
    makeup_img = Image.open(makeup_path).convert("RGB")

    image, face = inference.transfer(no_makeup_img, makeup_img, with_face=True)
    no_makeup_img_crop = \
        no_makeup_img.crop((face.left(), face.top(), face.right(), face.bottom()))
    image = postprocess(no_makeup_img_crop, image)

    if image.size[0] != no_makeup_img.size[0] or image.size[1] != no_makeup_img.size[1]:
        image = image.resize(no_makeup_img.size)

    image.save(result_path)

    return True
