"""Remove things that will not exist in sim (robot, people, cables, other cameras) from a real photo and
optionally outpaint beyond its borders, with SDXL inpainting. Needs a GPU, diffusers and the
SDXL inpainting weights (downloaded to HF_HOME on first use).

Pixels outside the (4-px feathered) mask are copied from the original at full resolution, so nothing
the camera actually saw gets blurred or resampled. Only the generated region comes from the diffusion
model, which runs at --work-width and is upsampled back.

Usage:
  python inpaint.py photo.jpg out.png --mask mask.png|polys.json --prompt "empty robotics lab, white walls, ..."
         [--pad T R B L] [--work-width 1024] [--seed 0] [--steps 40]
mask.png: white = regenerate. polys.json: [[[u, v], ...], ...] in photo pixels.
--pad: pixels of new content to add on each side (top right bottom left) at photo resolution.
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

NEGATIVE = "robot, robot arm, person, people, hands, blurry, distorted, text, watermark, cartoon, painting"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("out")
    ap.add_argument("--mask", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--pad", type=int, nargs=4, default=[0, 0, 0, 0])
    ap.add_argument("--work-width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    a = ap.parse_args()

    import torch
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import (
        StableDiffusionXLInpaintPipeline,
    )

    img = Image.open(a.photo).convert("RGB")
    T, R, B, L = a.pad
    W, H = img.width + L + R, img.height + T + B
    canvas = Image.new("RGB", (W, H), (128, 128, 128))
    canvas.paste(img, (L, T))
    mask = Image.new("L", (W, H), 0)
    if a.mask.endswith(".json"):
        d = ImageDraw.Draw(mask)
        for poly in json.load(open(a.mask)):
            d.polygon([(u + L, v + T) for u, v in poly], fill=255)
    else:
        mask.paste(Image.open(a.mask).convert("L").resize(img.size, Image.NEAREST), (L, T))
    d = ImageDraw.Draw(mask)
    for n, box in zip(a.pad, ((0, 0, W, T + 4), (W - R - 4, 0, W, H), (0, H - B - 4, W, H), (0, 0, L + 4, H))):
        if n > 0:
            d.rectangle(box, fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(9))

    s = a.work_width / W
    ww, wh = int(round(W * s / 8)) * 8, int(round(H * s / 8)) * 8
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(a.model, torch_dtype=torch.float16, variant="fp16").to("cuda")
    out = pipe(prompt=a.prompt, negative_prompt=a.negative, image=canvas.resize((ww, wh), Image.LANCZOS),
               mask_image=mask.resize((ww, wh), Image.NEAREST), width=ww, height=wh, strength=0.99,
               num_inference_steps=a.steps, guidance_scale=7.0,
               generator=torch.Generator("cuda").manual_seed(a.seed)).images[0]
    gen = np.asarray(out.resize((W, H), Image.LANCZOS), np.float32)
    m = np.asarray(mask.filter(ImageFilter.GaussianBlur(4)), np.float32)[..., None] / 255
    final = gen * m + np.asarray(canvas, np.float32) * (1 - m)
    Image.fromarray(np.clip(final, 0, 255).astype(np.uint8)).save(a.out)
    json.dump({"photo": a.photo, "pad_trbl": a.pad, "size": [W, H], "prompt": a.prompt, "seed": a.seed},
              open(a.out.rsplit(".", 1)[0] + "_inpaint.json", "w"), indent=1)
    print("[inpaint] wrote", a.out, (W, H), "generated fraction", round(float((m > 0.5).mean()), 3))


if __name__ == "__main__":
    main()
