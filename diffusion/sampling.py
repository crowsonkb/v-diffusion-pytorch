import torch
from tqdm.auto import trange

from . import utils

# These 4 sample_foo functions are subroutines called by sample()

def sample_step_pred(model, x, steps, eta, extra_args, ts, alphas, sigmas, i):
    # Get the model output (v, the predicted velocity)
    with torch.cuda.amp.autocast():
        v = model(x, ts * steps[i], **extra_args).float()

    # Predict the noise and the denoised image
    pred = x * alphas[i] - v * sigmas[i]

    return pred, v


def sample_step_noise(model, x, steps, eta, extra_args, ts, alphas, sigmas, i, pred, v):
    eps = x * sigmas[i] + v * alphas[i]

    # If we are not on the last timestep, compute the noisy image for the
    # next timestep.
    if i < len(steps) - 1:
        # If eta > 0, adjust the scaling factor for the predicted noise
        # downward according to the amount of additional noise to add
        ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
            (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
        adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

        # Recombine the predicted noise and predicted denoised image in the
        # correct proportions for the next step
        x = pred * alphas[i + 1] + eps * adjusted_sigma

        # Add the correct amount of fresh noise

        if eta:
            x = x + torch.randn_like(x) * ddim_sigma

    return x

def sample_setup(model, x, steps, eta, extra_args):
    """Draws samples from a model given starting noise."""

    # print("SAMPLE SETUP ", steps.shape)
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    sample_state = [model, steps, eta, extra_args, ts, alphas, sigmas]
    return sample_state

def sample_step(sample_state, x, i, last_pred, last_v):
    model, steps, eta, extra_args, ts, alphas, sigmas = sample_state
    pred, v = sample_step_pred(model, x, steps, eta, extra_args, ts, alphas, sigmas, i)
    return pred, v, x


def sample_noise(sample_state, x, i, last_pred, last_v):
    model, steps, eta, extra_args, ts, alphas, sigmas = sample_state
    if last_pred != None:
        x = sample_step_noise(model, x, steps, eta, extra_args, ts, alphas, sigmas, i, last_pred, last_v)
    return x

# this new version of sample calls the above four functions to do the work
def sample_split(model, x, steps, eta, extra_args):
    pred = None
    v = None
    sample_state = sample_setup(model, x, steps, eta, extra_args)
    for i in trange(len(steps)):
        pred, v, x = sample_step(sample_state, x, i, pred, v)
        x = sample_noise(sample_state, x, i, pred, v)

    return pred

# this is the original version of sample which did everything at once
@torch.no_grad()
def sample(model, x, steps, eta, extra_args, callback=None):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps)):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * steps[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # Call the callback
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'v': v, 'pred': pred})

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < len(steps) - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


@torch.no_grad()
def cond_sample(model, x, steps, eta, extra_args, cond_fn, callback=None):
    """Draws guided samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps)):

        # Get the model output
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            with torch.cuda.amp.autocast():
                v = model(x, ts * steps[i], **extra_args)

            pred = x * alphas[i] - v * sigmas[i]

            # Call the callback
            if callback is not None:
                callback({'x': x, 'i': i, 't': steps[i], 'v': v.detach(), 'pred': pred.detach()})

            if steps[i] < 1:
                cond_grad = cond_fn(x, ts * steps[i], pred, **extra_args).detach()
                v = v.detach() - cond_grad * (sigmas[i] / alphas[i])
            else:
                v = v.detach()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < len(steps) - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


@torch.no_grad()
def reverse_sample(model, x, steps, extra_args, callback=None):
    """Finds a starting latent that would produce the given image with DDIM
    (eta=0) sampling."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps) - 1):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * steps[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # Call the callback
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'v': v, 'pred': pred})

        # Recombine the predicted noise and predicted denoised image in the
        # correct proportions for the next step
        x = pred * alphas[i + 1] + eps * sigmas[i + 1]

    return x
