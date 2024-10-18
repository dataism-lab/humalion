from src.humalion.engine.video_generative_model import VideoGenerativeModel
from src.humalion.services.luma import LumaService


class LumaVideoGenModel(LumaService, VideoGenerativeModel):
    BASE_PROMPT_PREFIX = "The main person from the image is correctly saying the following text: "
    BASE_PROMPT_SUFFIX = " The camera should not move and be static. And the objects in the background should behave naturally."

    def prepare_model(self):
        pass

    def generate_speaking_head_video(self, source_img: str, text: str):
        # prompt = self.BASE_PROMPT_PREFIX + text + self.BASE_PROMPT_SUFFIX
        generation = self.client.generations.create(
            prompt=text,
            keyframes={
                "frame1": {
                    "type": "image",
                    "url": source_img
                },
            }
        )

        return generation
