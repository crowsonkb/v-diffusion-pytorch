import math
import io

import requests
import torch
from torchvision.transforms import functional as TF


def fetch(url_or_path):
    """Fetches a file from an HTTP or HTTPS url, or opens the local file."""
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def from_pil_image(x):
    """Converts from a PIL image to a tensor."""
    x = TF.to_tensor(x)
    if x.ndim == 2:
        x = x[..., None]
    return x * 2 - 1


def to_pil_image(x):
    """Converts from a tensor to a PIL image."""
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = x[0]
    if x.shape[0] == 1:
        x = x[0]
    return TF.to_pil_image((x.clamp(-1, 1) + 1) / 2)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def log_snr_to_alpha_sigma(log_snr):
    """Returns the scaling factors for the clean image and for the noise, given
    the log SNR for a timestep."""
    return log_snr.sigmoid().sqrt(), log_snr.neg().sigmoid().sqrt()


def alpha_sigma_to_log_snr(alpha, sigma):
    """Returns a log snr, given the scaling factors for the clean image and for
    the noise."""
    return torch.log(alpha**2 / sigma**2)


def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


def get_ddpm_schedule(ddpm_t):
    """Returns timesteps for the noise schedule from the DDPM paper."""
    log_snr = -torch.special.expm1(1e-4 + 10 * ddpm_t**2).log()
    alpha, sigma = log_snr_to_alpha_sigma(log_snr)
    return alpha_sigma_to_t(alpha, sigma)


def get_spliced_ddpm_cosine_schedule(t):
    """Returns timesteps for a spliced DDPM/cosine noise schedule."""
    ddpm_crossover = 0.48536712
    cosine_crossover = 0.80074257
    big_t = t * (1 + cosine_crossover - ddpm_crossover)
    ddpm_part = get_ddpm_schedule(big_t + ddpm_crossover - cosine_crossover)
    return torch.where(big_t < cosine_crossover, big_t, ddpm_part)


def get_log_schedule(t, min_log_snr=-10, max_log_snr=10):
    """Returns timesteps for a logarithmically spaced schedule."""
    log_snr = t * (min_log_snr - max_log_snr) + max_log_snr
    alpha, sigma = log_snr_to_alpha_sigma(log_snr)
    return alpha_sigma_to_t(alpha, sigma)
