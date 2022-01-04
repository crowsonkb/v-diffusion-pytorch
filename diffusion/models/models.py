from . import cc12m_1, yfcc_1, yfcc_2


models = {
    'cc12m_1': cc12m_1.CC12M1Model,
    'cc12m_1_cfg': cc12m_1.CC12M1Model,
    'yfcc_1': yfcc_1.YFCC1Model,
    'yfcc_2': yfcc_2.YFCC2Model,
}


def get_model(model):
    return models[model]


def get_models():
    return list(models.keys())
