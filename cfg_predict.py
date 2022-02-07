# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import cog
from pathlib import Path
from PIL import Image
import tempfile
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from CLIP import clip
from diffusion import get_model, sampling, utils


def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])


def parse_prompt(prompt, default_weight=3.):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', default_weight][len(vals):]
    return vals[0], float(vals[1])


class ClassifierFreeGuidanceDiffusionSampler(cog.Predictor):
    model_name = 'cc12m_1_cfg'
    checkpoint_path = 'checkpoints/cc12m_1_cfg.pth'
    device = 'cuda:0'

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        assert torch.cuda.is_available()
        self.model = get_model(self.model_name)()
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location='cpu'))
        self.model.half()
        self.model.to(self.device).eval().requires_grad_(False)
        self.clip = clip.load('ViT-B/16', jit=False, device=self.device)[0]
        self.clip.eval().requires_grad_(False)
        self.normalize_fn = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

    def normalize(self, image):
        return self.normalize_fn(image)


    def cfg_sample_fn(self, x, t, target_embeds, weights):
        n = x.shape[0]
        n_conds = len(target_embeds)
        x_in = x.repeat([n_conds, 1, 1, 1])
        t_in = t.repeat([n_conds])
        clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
        vs = self.model(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
        v = vs.mul(weights[:, None, None, None, None]).sum(0)
        return v

    
    def run_sampling(self, x, steps, eta):
        return sampling.sample(self.cfg_sample_fn, x, steps, eta, {})

    @cog.input('prompt', type=str, help='The prompt for image generation')
    @cog.input("eta", type=float, default=1.0, help='The amount of randomness')
    @cog.input('seed', type=int, default=0, help='Random seed for reproducibility.')
    @cog.input('steps', type=int, default=500, max=1000, min=0, help='Number of steps to sample for.')
    def predict(self, prompt: str, eta: float, seed: int, steps: int):
        """Run a single prediction on the model"""
        _, side_y, side_x = self.model.shape
        torch.manual_seed(seed)
        zero_embed = torch.zeros([1, clip.visual.output_dim], device=self.device)
        target_embeds, weights = [zero_embed], []
        txt, weight = parse_prompt(prompt)
        target_embeds.append(self.clip.encode_text(clip.tokenize(txt).to(self.device)).float())
        weights.append(weight)
        x = torch.randn([1, 3, side_y, side_x], device=self.device)
        t = torch.linspace(1, 0, steps + 1, device=self.device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        output_image = self.run_sampling(x, steps, eta)
        out_path = Path(tempfile.mkdtemp()) / "my-file.txt"
        utils.to_pil_image(output_image).save(out_path)
        return out_path
