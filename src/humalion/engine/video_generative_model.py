from abc import ABC, abstractmethod


class VideoGenerativeModel(ABC):
    @abstractmethod
    def generate_speaking_head_video(self, source_img: str) -> str:
        """
        Method must implement logic of generating a photo by prompt.
        Args:
            prompt: prompt for photo generating

        Returns:
            path to local photo generated by prompt
        """
        pass

    @abstractmethod
    def prepare_model(self):
        """
        Method that prepares the model for inference.
        You can create empty method if your model does not need to be prepared.
        """
        pass
