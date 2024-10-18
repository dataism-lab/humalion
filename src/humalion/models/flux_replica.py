from abc import ABC

from ..engine.image_generative_model import ImageGenerativeModel
from ..services.replica import ReplicaService
from ..utils.enum import StrEnum


class FluxReplica(ReplicaService, ImageGenerativeModel, ABC):
    output_dir = "output/flux_replica"

    class Models(StrEnum):
        DEV = "black-forest-labs/flux-dev"
        PRO = "black-forest-labs/flux-pro"
        SCHNELL = "black-forest-labs/flux-schnell"

    def prepare_model(self):
        pass

    def _generate_photo(self, prompt: str, model: str) -> dict:
        if model not in FluxReplica.Models:
            raise ValueError(f"Model {model} not supported")

        result = self.run_model(model=model, input_data={"prompt": prompt})
        return result


class FluxProReplica(FluxReplica):
    output_dir = "output/flux_pro_replica"

    def generate_photo(self, prompt: str) -> dict:
        return self._generate_photo(prompt, FluxProReplica.Models.PRO)


class FluxDevReplica(FluxReplica):
    output_dir = "output/flux_dev_replica"

    def generate_photo(self, prompt: str) -> dict:
        return self._generate_photo(prompt, FluxDevReplica.Models.DEV)


class FluxSchnellReplica(FluxReplica):
    output_dir = "output/flux_schnell_replica"

    def generate_photo(self, prompt: str) -> dict:
        return self._generate_photo(prompt, FluxSchnellReplica.Models.SCHNELL)
