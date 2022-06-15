#!/usr/bin/env python3

"""Applies a text prompt to an existing image by finding a latent that would produce it
with the unconditioned DDIM ODE, then integrating the text-conditional DDIM ODE starting
from that latent."""

import argparse
from functools import partial
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange

from CLIP import clip
from diffusion import get_model, get_models, sampling, utils

MODULE_DIR = Path(__file__).resolve().parent


def parse_prompt(prompt, default_weight=3.):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', default_weight][len(vals):]
    return vals[0], float(vals[1])


def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('init', type=str,
                   help='the init image')
    p.add_argument('prompts', type=str, default=[], nargs='*',
                   help='the text prompts to use')
    p.add_argument('--images', type=str, default=[], nargs='*', metavar='IMAGE',
                   help='the image prompts')
    p.add_argument('--checkpoint', type=str,
                   help='the checkpoint to use')
    p.add_argument('--device', type=str,
                   help='the device to use')
    p.add_argument('--max-timestep', '-mt', type=float, default=1.,
                   help='the maximum timestep')
    p.add_argument('--method', type=str, default='plms',
                   choices=['ddim', 'prk', 'plms', 'pie', 'plms2', 'iplms'],
                   help='the sampling method to use')
    p.add_argument('--model', type=str, default='cc12m_1_cfg', choices=['cc12m_1_cfg'],
                   help='the model to use')
    p.add_argument('--output', '-o', type=str, default='out.png',
                   help='the output filename')
    p.add_argument('--size', type=int, nargs=2,
                   help='the output image size')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of timesteps')
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = get_model(args.model)()
    _, side_y, side_x = model.shape
    if args.size:
        side_x, side_y = args.size
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = MODULE_DIR / f'checkpoints/{args.model}.pth'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    if device.type == 'cuda':
        model = model.half()
    model = model.to(device).eval().requires_grad_(False)
    clip_model_name = model.clip_model if hasattr(model, 'clip_model') else 'ViT-B/16'
    clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    init = Image.open(utils.fetch(args.init)).convert('RGB')
    init = resize_and_center_crop(init, (side_x, side_y))
    init = utils.from_pil_image(init).to(device)[None]

    zero_embed = torch.zeros([1, clip_model.visual.output_dim], device=device)
    target_embeds, weights = [zero_embed], []

    for prompt in args.prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in args.images:
        path, weight = parse_prompt(prompt)
        img = Image.open(utils.fetch(path)).convert('RGB')
        clip_size = clip_model.visual.input_resolution
        img = resize_and_center_crop(img, (clip_size, clip_size))
        batch = TF.to_tensor(img)[None].to(device)
        embed = F.normalize(clip_model.encode_image(normalize(batch)).float(), dim=-1)
        target_embeds.append(embed)
        weights.append(weight)

    weights = torch.tensor([1 - sum(weights), *weights], device=device)

    def cfg_model_fn(x, t):
        n = x.shape[0]
        n_conds = len(target_embeds)
        x_in = x.repeat([n_conds, 1, 1, 1])
        t_in = t.repeat([n_conds])
        clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
        vs = model(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
        v = vs.mul(weights[:, None, None, None, None]).sum(0)
        return v

    def run():
        t = torch.linspace(0, 1, args.steps + 1, device=device)
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        steps = steps[steps <= args.max_timestep]
        if args.method == 'ddim':
            x = sampling.reverse_sample(model, init, steps, {'clip_embed': zero_embed})
            out = sampling.sample(cfg_model_fn, x, steps.flip(0)[:-1], 0, {})
        if args.method == 'prk':
            x = sampling.prk_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.prk_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        if args.method == 'plms':
            x = sampling.plms_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.plms_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        if args.method == 'pie':
            x = sampling.pie_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.pie_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        if args.method == 'plms2':
            x = sampling.plms2_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.plms2_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        if args.method == 'iplms':
            x = sampling.iplms_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.iplms_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        utils.to_pil_image(out[0]).save(args.output)

    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
