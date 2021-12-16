#!/usr/bin/env python3

"""CLIP guided sampling from a diffusion model."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm import trange

from CLIP import clip
from diffusion import get_model, get_models, sampling, utils

MODULE_DIR = Path(__file__).resolve().parent


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.adaptive_avg_pool2d(cutout, self.cut_size)
            cutouts.append(cutout)
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('prompt', type=str,
                   help='the text prompt')
    p.add_argument('--batch-size', '-bs', type=int, default=1,
                   help='the number of images per batch')
    p.add_argument('--checkpoint', type=str,
                   help='the checkpoint to use')
    p.add_argument('--clip-guidance-scale', '-cs', type=float, default=500.,
                   help='the CLIP guidance scale')
    p.add_argument('--device', type=str,
                   help='the device to use')
    p.add_argument('--eta', type=float, default=1.,
                   help='the amount of noise to add during sampling (0-1)')
    p.add_argument('--model', type=str, default='cc12m_1', choices=get_models(),
                   help='the model to use')
    p.add_argument('-n', type=int, default=1,
                   help='the number of images to sample')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    p.add_argument('--steps', type=int, default=1000,
                   help='the number of timesteps')
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = get_model(args.model)()
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = MODULE_DIR / f'checkpoints/{args.model}.pth'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    if device.type == 'cuda':
        model = model.half()
    model = model.to(device).eval().requires_grad_(False)
    clip_model = clip.load(model.clip_model, jit=False, device=device)[0]
    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    cutn = 16
    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, cutn=cutn, cut_pow=1)

    clip_embed = clip_model.encode_text(clip.tokenize(args.prompt).to(device))
    clip_embed = clip_embed.repeat([args.n, 1])

    torch.manual_seed(args.seed)

    def cond_fn(x, t, pred, clip_embed):
        clip_in = normalize(make_cutouts((pred + 1) / 2))
        image_embeds = clip_model.encode_image(clip_in).view([cutn, x.shape[0], -1])
        losses = spherical_dist_loss(image_embeds, clip_embed[None])
        loss = losses.mean(0).sum() * args.clip_guidance_scale
        grad = -torch.autograd.grad(loss, x)[0]
        return grad

    def run(x, clip_embed):
        t = torch.linspace(1, 0, args.steps + 1, device=device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        extra_args = {'clip_embed': clip_embed}
        if not args.clip_guidance_scale:
            return sampling.sample(model, x, steps, args.eta, extra_args)
        return sampling.cond_sample(model, x, steps, args.eta, extra_args, cond_fn)

    def run_all(n, batch_size):
        x = torch.randn([args.n, *model.shape], device=device)
        for i in trange(0, n, batch_size):
            cur_batch_size = min(n - i, batch_size)
            outs = run(x[i:i+cur_batch_size], clip_embed[i:i+cur_batch_size])
            for j, out in enumerate(outs):
                utils.to_pil_image(out).save(f'out_{i + j:05}.png')

    try:
        run_all(args.n, args.batch_size)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
