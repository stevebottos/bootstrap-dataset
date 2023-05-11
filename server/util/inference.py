from io import BytesIO

import clip
import numpy as np
import pandas as pd
import validators
import torch
from PIL import Image

import urllib.request
import traceback
import io

from .resizer import Resizer


# From https://github.com/rom1504/img2dataset/blob/main/img2dataset/downloader.py
def download_image(
    url,
    timeout=15,
    user_agent_token=None,
    disallowed_header_directives=["noai", "noindex"],
):
    """Download an image with urllib"""

    def _is_disallowed(headers, user_agent_token, disallowed_header_directives):
        """Check if HTTP headers contain an X-Robots-Tag directive disallowing usage"""
        for values in headers.get_all("X-Robots-Tag", []):
            try:
                uatoken_directives = values.split(":", 1)
                directives = [
                    x.strip().lower() for x in uatoken_directives[-1].split(",")
                ]
                ua_token = (
                    uatoken_directives[0].lower()
                    if len(uatoken_directives) == 2
                    else None
                )
                if (ua_token is None or ua_token == user_agent_token) and any(
                    x in disallowed_header_directives for x in directives
                ):
                    return True
            except Exception as err:  # pylint: disable=broad-except
                traceback.print_exc()
                print(f"Failed to parse X-Robots-Tag: {values}: {err}")
        return False

    img_stream = None
    user_agent_string = (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    )
    if user_agent_token:
        user_agent_string += f" (compatible; {user_agent_token}; +https://github.com/rom1504/img2dataset)"
    try:
        request = urllib.request.Request(
            url, data=None, headers={"User-Agent": user_agent_string}
        )
        with urllib.request.urlopen(request, timeout=timeout) as r:
            if disallowed_header_directives and _is_disallowed(
                r.headers,
                user_agent_token,
                disallowed_header_directives,
            ):
                return None, "Use of image disallowed by X-Robots-Tag directive"
            img_stream = io.BytesIO(r.read())
        return img_stream, None
    except Exception as err:  # pylint: disable=broad-except
        if img_stream is not None:
            img_stream.close()
        return None, str(err)


def inference(url_or_filepath):
    # Pre-Preprocess
    try:
        resizer = Resizer(224, "border", False)
        if validators.url(url_or_filepath):
            image_bytes, err = download_image(url_or_filepath)
        else:
            with open(url_or_filepath, "rb") as image:
                image_bytes = BytesIO(image.read())
        image_bytes = BytesIO(resizer(image_bytes)[0])
        image = Image.open(image_bytes, formats=["JPEG"])
    except:
        return (
            None,
            f"The url {url_or_filepath} failed. Are you sure you have the right link?",
        )

    # Inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        clip_embedding = model.encode_image(image)
        clip_embedding /= clip_embedding.norm(dim=-1, keepdim=True)
    clip_embedding = clip_embedding.cpu().numpy()

    return clip_embedding, None


if __name__ == "__main__":
    # http://media.rightmove.co.uk/dir/87k/86030/41964130/86030_4197229_IMG_00_0000_max_214x143.jpg
    # test.jpg
    inference(
        "http://media.rightmove.co.uk/dir/87k/86030/41964130/86030_4197229_IMG_00_0000_max_214x143.jpg"
    )
