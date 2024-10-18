import glob
import os
import os.path as osp
import zipfile

import onnxruntime
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from insightface.utils import DEFAULT_MP_NAME
from insightface.utils.download import download_file
from insightface.utils.storage import BASE_REPO_URL
from loguru import logger


def download(sub_dir, name, force=False, root="~/.insightface"):
    _root = os.path.expanduser(root)
    dir_path = os.path.join(_root, sub_dir, name)
    if osp.exists(dir_path) and not force:
        return dir_path
    zip_file_path = os.path.join(_root, sub_dir, name + ".zip")
    model_url = "%s/%s.zip" % (BASE_REPO_URL, name)
    download_file(model_url, path=zip_file_path, overwrite=True)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(os.path.join(_root, sub_dir))
    os.remove(zip_file_path)
    return dir_path


def ensure_available(sub_dir, name, root="~/.insightface"):
    return download(sub_dir, name, force=False, root=root)


class FixedFaceAnalysis(FaceAnalysis):
    def __init__(self, name=DEFAULT_MP_NAME, root="~/.insightface", allowed_modules=None, **kwargs):  # noqa
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_dir = ensure_available("models", name, root=root)
        onnx_files = glob.glob(osp.join(self.model_dir, "*.onnx"))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                logger.warning("model not recognized:", onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                logger.warning("model ignore:", onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                logger.warning(
                    "find model:", onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std
                )
                self.models[model.taskname] = model
            else:
                logger.warning("duplicated model task type, ignore:", onnx_file, model.taskname)
                del model
        assert "detection" in self.models
        self.det_model = self.models["detection"]
