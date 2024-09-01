from abc import ABC, abstractmethod
from enum import Enum
from faker import Faker
import random


class ABSPersona(ABC):
    @abstractmethod
    def prompt(self) -> str:
        pass


class Gender(str, Enum):
    MALE = 'male'
    FEMALE = 'female'


class SkinTone(str, Enum):
    PORCLEAN = 'porclean'
    IVORY = 'ivory'
    WARM_IVORY = 'warm_ivory'
    SAND = 'sand'
    ROSE_BEGE = 'rose_bege'
    NATURAL = 'natural'
    BEGE = 'bege'
    SENA = 'sena'
    BAND = 'band'
    ALMOND = 'almond'
    AMBER = 'amber'
    UMBER = 'umber'
    BRONZE = 'bronze'
    ESPRESSO = 'espresso'
    CHOCOLATE = 'chocolate'


class Beard(str, Enum):
    CLEAN_SHAVED = 'clean shaved'
    CIRCLE_BEARD = 'A chin patch and a mustache that forms a circle'
    ROYALE_BEARD = 'A mustache anchored by a chin strip'
    GOATEE = 'A small beard that elongates the chin'
    PETITE_GOATEE = 'A small beard that elongates the chin'
    VAN_DYKE_BEARD = 'A full goatee with detached mustache'
    SHORT_BOXED_BEARD = 'A short beard with thin, neatly trimmed sides'
    BALBO_BEARD = 'A beard without sideburns and a trimmed, floating mustache'
    ANCHOR_BEARD = 'A pointed beard that traces the jawline, paired with a mustache'
    CHEVRON = 'A mustache that covers your entire top lip'
    THREE_DAY_BEARD = 'A closely trimmed beard that simulates 3 days of stubble'
    HORSESHOE_MUSTACHE = 'A mustache with long bars pointing downward'
    ORIGINAL_STACHE = 'A trim mustache that sits just above the top lip'
    MUTTON_CHOPS_BEARD = 'Long sideburns that connect to a mustache'
    GUNSLINGER_BEARD_AND_MUSTACHE = 'Flared sideburns paired with a horseshoe mustache'
    CHIN_STRIP = 'A vertical line of hair across the chin'
    CHIN_STRAP_STYLE_BEARD = 'A beard with no mustache that circles the chin'


class FaceShape(str, Enum):
    OVAL = "oval"
    ROUND = "round"
    SQUARE = "square"
    DIAMOND = "diamond"
    HEART = "heart"
    RECTANGLE = "rectangle"


class BodyShape(Enum):
    """
    TRIANGLE: This body shape typically has wider hips and shoulders that are narrower in comparison.
    COLUMN: This body shape has hardly any difference in width between the shoulders, waist, or hips.
    RECTANGLE: This body shape initially appears to be column-shaped, but is more athletic aesthetically as the hips and shoulders are equal width.
    OVAL: This body shape typically has wider waist and chest areas than the hips.
    INVERTED_TRIANGLE: This body shape features broad shoulders with slim waist and hips.
    HOURGLASS: This body shape is known for its balanced bust and hip proportions with a slender waist.
    FULL_HOURGLASS: This is similar to the hourglass shape, but it typically features a fuller bust.
    """
    TRIANGLE = "triangle"
    COLUMN = "column"
    RECTANGLE = "rectangle"
    OVAL = "oval"
    INVERTED_TRIANGLE = "inverted triangle"
    HOURGLASS = "hourglass"
    FULL_HOURGLASS = "full hourglass"


class Persona(ABSPersona):
    """
    A class representing a persona by parameters.

    Attributes:
        name (str): The name of the persona.
        gender (Gender): The gender of the persona.
        skintone (SkinTone): The skin tone of the persona.
        beard (Beard): The presence of beard on the persona.
        hair_color (str): The hair color of the persona.
        hair_style (str): The hairstyle of the persona.
        face_shape (FaceShape): The face shape of the persona.
        body_shape (BodyShape): The body shape of the persona.
        height (int): The height of the persona in centimeters.
        clothes (str): The type of clothes worn by the persona.
        additional_parameters (dict): Additional parameters specific to the persona.
        random_person (bool): generate a random persona.

    Methods:
        prompt(): Returns the full prompt for the persona.

    """
    def __init__(
            self,
            name: str | None = None,
            gender: Gender | None = None,
            age: int | None = None,
            skintone: SkinTone | None = None,
            beard: Beard = Beard.CLEAN_SHAVED,
            hair_color: str = 'blonde',
            hair_style: str = '',
            face_shape: FaceShape = FaceShape.DIAMOND,
            body_shape: BodyShape = BodyShape.COLUMN,
            height: int = 180,
            clothes: str = '',
            race: str | None = None,
            additional_parameters: dict | None = None,
            random_person: bool = False,
    ):
        if random_person:
            faker = Faker()
            name = faker.name()
            gender = random.choice(list(Gender))
            age = random.randint(12, 90)
            skintone = random.choice(list(SkinTone))
            beard = random.choice(list(Beard))
            height = random.randint(150, 210)
        else:
            if not all((name, gender, age)):
                raise ValueError("All mandatory parameters must be given (name, gender, age).")

        if age <= 0:
            raise ValueError("Age must be positive")

        self.name = name
        self.gender = gender
        self.age = age
        self.skintone = skintone
        self.beard = beard
        self.hair_color = hair_color
        self.hair_style = hair_style
        self.face_shape = face_shape
        self.body_shape = body_shape
        self.height = height
        self.clothes = clothes
        self.race = race
        self.additional_parameters = additional_parameters

    def _gender_with_age(self):
        prompt = "person"
        if self.gender == Gender.MALE and 0 < self.age < 3:
            prompt = "baby boy"
        elif self.gender == Gender.FEMALE and 0 < self.age < 3:
            prompt = "baby girl"
        elif self.gender == Gender.MALE and 3 <= self.age < 12:
            prompt = "boy"
        elif self.gender == Gender.FEMALE and 3 <= self.age < 12:
            prompt = "girl"
        elif self.gender == Gender.MALE and 12 <= self.age < 18:
            prompt = "teenage boy"
        elif self.gender == Gender.FEMALE and 12 <= self.age < 18:
            prompt = "teenage girl"
        elif self.gender == Gender.MALE and 18 <= self.age < 23:
            prompt = "young girl"
        elif self.gender == Gender.FEMALE and 18 <= self.age < 23:
            prompt = "young man"
        elif self.gender == Gender.MALE and 23 <= self.age < 60:
            prompt = "man"
        elif self.gender == Gender.FEMALE and 18 <= self.age < 30:
            prompt = "girl"
        elif self.gender == Gender.FEMALE and 30 <= self.age < 70:
            prompt = "woman"
        elif self.gender == Gender.MALE and self.age > 60:
            prompt = "old man"
        elif self.gender == Gender.FEMALE and self.age > 70:
            prompt = "old woman"

        prompt += f" of {self.age} years old"
        return prompt

    def prompt(self) -> str:
        prompt = self.race if self.race is not None else ""
        prompt += self._gender_with_age()
        if self.skintone:
            prompt += f" with {self.skintone.value} skin tone, "
        if self.beard != Beard.CLEAN_SHAVED:
            prompt += f"{self.beard.value} beard, "

        prompt += f"{self.hair_color} hair color, "
        if self.hair_style:
            prompt += f"{self.hair_style} hairstyle, "
        prompt += f"{self.face_shape.value} face shape, "
        prompt += f"{self.body_shape.value} body shape, "
        prompt += f"{self.height} cm tall, "
        if self.clothes:
            prompt += f"wearing {self.clothes}, "
        if self.additional_parameters:
            prompt += ". "
            for key, value in self.additional_parameters.items():
                prompt += f"{key} is {value}, "

        return prompt
