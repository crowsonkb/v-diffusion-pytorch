# v-diffusion-pytorch

v objective diffusion inference code for PyTorch, by Katherine Crowson ([@RiversHaveWings](https://twitter.com/RiversHaveWings)) and Chainbreakers AI ([@jd_pressman](https://twitter.com/jd_pressman)).

The models are denoising diffusion probabilistic models (https://arxiv.org/abs/2006.11239), which are trained to reverse a gradual noising process, allowing the models to generate samples from the learned data distributions starting from random noise. DDIM-style deterministic sampling (https://arxiv.org/abs/2010.02502) is also supported. The models are also trained on continuous timesteps. They use the 'v' objective from Progressive Distillation for Fast Sampling of Diffusion Models (https://openreview.net/forum?id=TIdIXIpzhoI).

Thank you to [stability.ai](https://www.stability.ai) for compute to train these models!

## Dependencies

- PyTorch ([installation instructions](https://pytorch.org/get-started/locally/))

- requests, tqdm (install with `pip install`)

- CLIP (https://github.com/openai/CLIP), and its additional pip-installable dependencies: ftfy, regex. **If you `git clone --recursive` this repo, it should fetch CLIP automatically.**

## Model checkpoints:

- [CC12M 256x256](https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1.pth), SHA-256 `63946d1f6a1cb54b823df818c305d90a9c26611e594b5f208795864d5efe0d1f`

A 602M parameter CLIP conditioned model trained on [Conceptual 12M](https://github.com/google-research-datasets/conceptual-12m) for 3.1M steps.

## Sampling

### Example

If the model checkpoints are stored in `checkpoints/`, the following will generate an image:

```
./clip_sample.py "the rise of consciousness" --model cc12m_1 --seed 0
```

If they are somewhere else, you need to specify the path to the checkpoint with `--checkpoint`.

### CLIP conditioned/guided sampling

```
usage: clip_sample.py [-h] [--images [IMAGE ...]] [--batch-size BATCH_SIZE]
                      [--checkpoint CHECKPOINT] [--clip-guidance-scale CLIP_GUIDANCE_SCALE]
                      [--device DEVICE] [--eta ETA] [--model {cc12m_1}] [-n N] [--seed SEED]
                      [--steps STEPS]
                      [prompts ...]
```

`prompts`: the text prompts to use. Relative weights for text prompts can be specified by putting the weight after a colon, for example: `"the rise of consciousness:0.5"`.

`--batch-size`: sample this many images at a time (default 1)

`--checkpoint`: manually specify the model checkpoint file

`--clip-guidance-scale`: how strongly the result should match the text prompt (default 500). If set to 0, the cc12m_1 model will still be CLIP conditioned and sampling will go faster and use less memory.

`--device`: the PyTorch device name to use (default autodetects)

`--eta`: set to 0 for deterministic (DDIM) sampling, 1 (the default) for stochastic (DDPM) sampling, and in between to interpolate between the two. DDIM is preferred for low numbers of timesteps.

`--images`: the image prompts to use (local files or HTTP(S) URLs). Relative weights for image prompts can be specified by putting the weight after a colon, for example: `"image_1.png:0.5"`.

`--model`: specify the model to use (default cc12m_1)

`-n`: sample until this many images are sampled (default 1)

`--seed`: specify the random seed (default 0)

`--steps`: specify the number of diffusion timesteps (default is 1000, can lower for faster but lower quality sampling)
