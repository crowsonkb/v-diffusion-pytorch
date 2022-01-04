# v-diffusion-pytorch

v objective diffusion inference code for PyTorch, by Katherine Crowson ([@RiversHaveWings](https://twitter.com/RiversHaveWings)) and Chainbreakers AI ([@jd_pressman](https://twitter.com/jd_pressman)).

The models are denoising diffusion probabilistic models (https://arxiv.org/abs/2006.11239), which are trained to reverse a gradual noising process, allowing the models to generate samples from the learned data distributions starting from random noise. DDIM-style deterministic sampling (https://arxiv.org/abs/2010.02502) is also supported. The models are also trained on continuous timesteps. They use the 'v' objective from Progressive Distillation for Fast Sampling of Diffusion Models (https://openreview.net/forum?id=TIdIXIpzhoI). Guided diffusion sampling scripts (https://arxiv.org/abs/2105.05233) are included, specifically CLIP guided diffusion. This repo also includes a diffusion model conditioned on CLIP text embeddings that supports classifier-free guidance (https://openreview.net/pdf?id=qw8AKxfYbI), similar to GLIDE (https://arxiv.org/abs/2112.10741).

Thank you to [stability.ai](https://www.stability.ai) for compute to train these models!

## Dependencies

- PyTorch ([installation instructions](https://pytorch.org/get-started/locally/))

- requests, tqdm (install with `pip install`)

- CLIP (https://github.com/openai/CLIP), and its additional pip-installable dependencies: ftfy, regex. **If you `git clone --recursive` this repo, it should fetch CLIP automatically.**

## Model checkpoints:

- [CC12M_1 CFG 256x256](https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1_cfg.pth), SHA-256 `4fc95ee1b3205a3f7422a07746383776e1dbc367eaf06a5b658ad351e77b7bda`

A 602M parameter CLIP conditioned model trained on [Conceptual 12M](https://github.com/google-research-datasets/conceptual-12m) for 3.1M steps and then fine-tuned for classifier-free guidance for 250K additional steps. **This is the recommended model to use.**

- [CC12M_1 256x256](https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1.pth), SHA-256 `63946d1f6a1cb54b823df818c305d90a9c26611e594b5f208795864d5efe0d1f`

As above, before CFG fine-tuning. The model from the original release of this repo.

- [YFCC_1 512x512](https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_1.pth), SHA-256 `a1c0f6baaf89cb4c461f691c2505e451ff1f9524744ce15332b7987cc6e3f0c8`

A 481M parameter unconditional model trained on a 33 million image original resolution subset of [Yahoo Flickr Creative Commons 100 Million](http://projects.dfki.uni-kl.de/yfcc100m/).

- [YFCC_2 512x512](https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_2.pth), SHA-256 `69ad4e534feaaebfd4ccefbf03853d5834231ae1b5402b9d2c3e2b331de27907`

A 968M parameter unconditional model trained on a 33 million image original resolution subset of [Yahoo Flickr Creative Commons 100 Million](http://projects.dfki.uni-kl.de/yfcc100m/).

## Sampling

### Example

If the model checkpoint for cc12m_1_cfg is stored in `checkpoints/`, the following will generate four images:

```
./cfg_sample.py "the rise of consciousness":5 -n 4 -bs 4 --seed 0
```

If they are somewhere else, you need to specify the path to the checkpoint with `--checkpoint`.

### CFG sampling (best, but only cc12m_1_cfg supports it)

```
usage: cfg_sample.py [-h] [--images [IMAGE ...]] [--batch-size BATCH_SIZE]
                     [--checkpoint CHECKPOINT] [--device DEVICE] [--eta ETA] [--init INIT]
                     [--model {cc12m_1_cfg}] [-n N] [--seed SEED] [--size SIZE SIZE]
                     [--starting-timestep STARTING_TIMESTEP] [--steps STEPS]
                     [prompts ...]
```

`prompts`: the text prompts to use. Weights for text prompts can be specified by putting the weight after a colon, for example: `"the rise of consciousness:5"`. A weight of 1 will sample images that match the prompt roughly as well as images usually match prompts like that in the training set. The default weight is 3.

`--batch-size`: sample this many images at a time (default 1)

`--checkpoint`: manually specify the model checkpoint file

`--device`: the PyTorch device name to use (default autodetects)

`--eta`: set to 0 for deterministic (DDIM) sampling, 1 (the default) for stochastic (DDPM) sampling, and in between to interpolate between the two. DDIM is preferred for low numbers of timesteps.

`--images`: the image prompts to use (local files or HTTP(S) URLs). Weights for image prompts can be specified by putting the weight after a colon, for example: `"image_1.png:5"`. The default weight is 3.

`--init`: specify the init image (optional)

`--model`: specify the model to use (default cc12m_1_cfg)

`-n`: sample until this many images are sampled (default 1)

`--seed`: specify the random seed (default 0)

`--starting-timestep`: specify the starting timestep if an init image is used (range 0-1, default 0.9)

`--size`: the output image size (default auto)

`--steps`: specify the number of diffusion timesteps (default is 500, can lower for faster but lower quality sampling)


### CLIP guided sampling (all models)

```
usage: clip_sample.py [-h] [--images [IMAGE ...]] [--batch-size BATCH_SIZE]
                      [--checkpoint CHECKPOINT] [--clip-guidance-scale CLIP_GUIDANCE_SCALE]
                      [--cutn CUTN] [--cut-pow CUT_POW] [--device DEVICE] [--eta ETA]
                      [--init INIT] [--model {cc12m_1,yfcc_1,yfcc_2}] [-n N] [--seed SEED]
                      [--size SIZE SIZE] [--starting-timestep STARTING_TIMESTEP] [--steps STEPS]
                      [prompts ...]
```

`prompts`: the text prompts to use. Relative weights for text prompts can be specified by putting the weight after a colon, for example: `"the rise of consciousness:0.5"`.

`--batch-size`: sample this many images at a time (default 1)

`--checkpoint`: manually specify the model checkpoint file

`--clip-guidance-scale`: how strongly the result should match the text prompt (default 500). If set to 0, the cc12m_1 model will still be CLIP conditioned and sampling will go faster and use less memory.

`--cutn`: the number of random crops to compute CLIP embeddings for (default 16)

`--cut-pow`: the random crop size power (default 1)

`--device`: the PyTorch device name to use (default autodetects)

`--eta`: set to 0 for deterministic (DDIM) sampling, 1 (the default) for stochastic (DDPM) sampling, and in between to interpolate between the two. DDIM is preferred for low numbers of timesteps.

`--images`: the image prompts to use (local files or HTTP(S) URLs). Relative weights for image prompts can be specified by putting the weight after a colon, for example: `"image_1.png:0.5"`.

`--init`: specify the init image (optional)

`--model`: specify the model to use (default cc12m_1)

`-n`: sample until this many images are sampled (default 1)

`--seed`: specify the random seed (default 0)

`--starting-timestep`: specify the starting timestep if an init image is used (range 0-1, default 0.9)

`--size`: the output image size (default auto)

`--steps`: specify the number of diffusion timesteps (default is 1000, can lower for faster but lower quality sampling)
