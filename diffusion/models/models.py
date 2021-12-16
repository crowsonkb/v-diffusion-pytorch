from . import cc12m_1


models = {
    'cc12m_1': cc12m_1.CC12M1Model,
}


def get_model(model):
    return models[model]


def get_models():
    return list(models.keys())
