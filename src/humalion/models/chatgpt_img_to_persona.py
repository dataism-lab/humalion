import base64

import instructor
from openai import OpenAI

from src.humalion.engine.img_to_persona_model import ImgToPersonaModel
from src.humalion.engine.persona import Persona


class ChatGPT4ImgToPersona(ImgToPersonaModel):
    OPENAI_RESPONSE_TIMEOUT_SECONDS = 25
    REQUEST_PROMPT = ("You are portrait painter. Given the image of the person your goal is to "
                      "carefully describe its appearance. Make description as exact as you can.")
    MODEL_NAME = "gpt-4o-2024-05-13"

    def __init__(self, openai_api_key: str):
        self.model = OpenAI(api_key=openai_api_key)
        self.instructor = instructor.from_openai(self.model)

    def get_persona(self, image_path: str) -> Persona:
        with open(image_path, "rb") as image_file:
            img_obj = image_file.read()

        encoded_image = base64.b64encode(img_obj).decode("utf-8")
        data = [
            {
                "role": "user", "content": [
                {
                    "type": "text",
                    "text": self.REQUEST_PROMPT,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
            }
        ]
        resp = self.instructor.chat.completions.create(
            model=self.MODEL_NAME, response_model=Persona, messages=data, max_retries=2
        )
        return resp

    def prepare_model(self):
        pass
