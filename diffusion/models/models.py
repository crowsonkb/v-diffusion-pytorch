from . import cc12m_1, danbooru_128, imagenet_128, wikiart_128, wikiart_256, yfcc_1, yfcc_2


models = {
    'cc12m_1': cc12m_1.CC12M1Model,
    'cc12m_1_cfg': cc12m_1.CC12M1Model,
    'danbooru_128': danbooru_128.Danbooru128Model,
    'imagenet_128': imagenet_128.ImageNet128Model,
    'wikiart_128': wikiart_128.WikiArt128Model,
    'wikiart_256': wikiart_256.WikiArt256Model,
    'yfcc_1': yfcc_1.YFCC1Model,
    'yfcc_2': yfcc_2.YFCC2Model,
}


def get_model(model):
    return models[model]


def get_models():
    return list(models.keys())
