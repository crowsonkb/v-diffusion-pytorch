from . import cc12m_1, yfcc_1, yfcc_2
import requests
import shutil
import os

models = {
    'cc12m_1': cc12m_1.CC12M1Model,
    'cc12m_1_cfg': cc12m_1.CC12M1Model,
    'yfcc_1': yfcc_1.YFCC1Model,
    'yfcc_2': yfcc_2.YFCC2Model,
}

model_to_url = {
    "yfcc_2":"https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_2.pth",
    "yfcc_1":"https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_1.pth",
    "cc12m_1":"https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1.pth",
    "cc12m_1_cfg":"https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1_cfg.pth",
}


def get_model(model):
    return models[model]


def get_models():
    return list(models.keys())



def download_model(model_name, file_path=None):
    model_url = model_to_url[model_name]
    if file_path is None:
        file_path = f"checkpoints/{model_name}.pth"
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with requests.get(model_url, stream=True) as r:
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

