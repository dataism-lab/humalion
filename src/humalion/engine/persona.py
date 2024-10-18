from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable

from faker import Faker
from pydantic import BaseModel, Field, PositiveInt

from ..utils.enum import StrEnum


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"


class ABSPersona(ABC):
    name: str
    gender: Gender

    @abstractmethod
    def prompt(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def mean(cls, people: Iterable["ABSPersona"]) -> "ABSPersona":
        pass


class SkinTone(StrEnum):
    PORCLEAN = "porclean"
    IVORY = "ivory"
    WARM_IVORY = "warm_ivory"
    SAND = "sand"
    ROSE_BEGE = "rose_bege"
    NATURAL = "natural"
    BEGE = "bege"
    SENA = "sena"
    BAND = "band"
    ALMOND = "almond"
    AMBER = "amber"
    UMBER = "umber"
    BRONZE = "bronze"
    ESPRESSO = "espresso"
    CHOCOLATE = "chocolate"


class Beard(StrEnum):
    CLEAN_SHAVED = "clean shaved"
    CIRCLE_BEARD = "circle beard: chin patch and a mustache that forms a circle"
    ROYALE_BEARD = "royale beard: mustache anchored by a chin strip"
    GOATEE = "goatee beard: small beard that elongates the chin"
    PETITE_GOATEE = "petite goatee beard: small beard that elongates the chin"
    VAN_DYKE_BEARD = "Van Dyke beard: full goatee with detached mustache"
    SHORT_BOXED_BEARD = "short boxed beard: short beard with thin, neatly trimmed sides"
    BALBO_BEARD = "balbo beard: beard without sideburns and a trimmed, floating mustache"
    ANCHOR_BEARD = "anchor beard: pointed beard that traces the jawline, paired with a mustache"
    CHEVRON = "chevron beard: mustache that covers your entire top lip"
    THREE_DAY_BEARD = "three days beard: closely trimmed beard that simulates 3 days of stubble"
    HORSESHOE_MUSTACHE = "horseshoe mustache: mustache with long bars pointing downward"
    ORIGINAL_STACHE = "original stache: trim mustache that sits just above the top lip"
    MUTTON_CHOPS_BEARD = "mutton chops beard: long sideburns that connect to a mustache"
    GUNSLINGER_BEARD_AND_MUSTACHE = "gunslinger beard and mustache: flared sideburns paired with a horseshoe mustache"
    CHIN_STRIP = "chin strip: vertical line of hair across the chin"
    CHIN_STRAP_STYLE_BEARD = "chin strap style beard: beard with no mustache that circles the chin"


class FaceShape(StrEnum):
    OVAL = "oval"
    ROUND = "round"
    SQUARE = "square"
    DIAMOND = "diamond"
    HEART = "heart"
    RECTANGLE = "rectangle"


class BodyShape(StrEnum):
    """
    TRIANGLE: This body shape typically has wider hips and shoulders that are narrower in comparison.
    COLUMN: This body shape has hardly any difference in width between the shoulders, waist, or hips.
    RECTANGLE: This body shape initially appears to be column-shaped,
        but is more athletic aesthetically as the hips and shoulders are equal width.
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


class EyesColor(StrEnum):
    BLUE = "blue"
    BROWN = "brown"
    GREEN = "green"
    GREY = "grey"


class HairCut(StrEnum):
    BALD = "no hair, bald"
    SHORT = "short hairstyle"
    MEDIUMSTRAIGHT = "medium straight hair"
    MEDIUMCURLY = "medium curly hair"
    MEDIUMWAVY = "medium wavy hair"
    LONGSTRAIGHT = "long straight hair"
    LONGCURLY = "long curly hair"
    LONGWAVY = "long wavy hair"
    PONYTAIL = "ponytail"


class ViewType(StrEnum):
    PORTRAIT = "portrait view"
    HADNSHOULDERS = "head and shoulders view"
    HALFBODY = "upper body portrait view"
    FULLBODY = "full body view"


class HairColor(StrEnum):
    BALD = ""
    BLACK = "black hair"
    BROWN = "brown hair"
    BLOND = "blond hair"
    RED = "red hair"
    GREY = "grey hair"
    WHITE = "white hair"


class Face(BaseModel):
    """
    Attributes:
        beard (Beard): The presence of beard on the persona.
        hair_color (str): The hair color of the persona.
        hair_style (str): The hairstyle of the persona.
        face_shape (FaceShape): The face shape of the persona.
        eyes_color (EyesColor): The eyes color of the persona.
    """

    beard: Beard | str | None = Field(
        description="Correctly assign beard of person on the image", default=Beard.CLEAN_SHAVED
    )
    hair_color: HairColor | str | None = Field(
        description="Correctly assign hair color of person on the image", default=None
    )
    hair_style: HairCut | str | None = Field(
        description="Correctly assign haircut type of person on the image or describe it by yourself", default=None
    )
    face_shape: FaceShape | str = Field(
        description="Correctly assign face shape of person on the image", default=FaceShape.DIAMOND
    )
    eyes_color: EyesColor | str | None = Field(
        description="Correctly assign eyes color of person on the image", default=None
    )


class Body(BaseModel):
    """
    Attributes:
        height (int): The height of the persona in centimeters.
        body_shape (BodyShape): The body shape of the persona.
        view (ViewType): The view type of the persona.
        clothes (str): The type of clothes worn by the persona.
    """

    height: PositiveInt | None = Field(
        description="Correctly assign body height of person on the image in centimeters", default=None
    )
    view: ViewType | None = Field(description="Correctly assign view of person on the image")
    body_shape: BodyShape | None = Field(description="Correctly assign body shape of person on the image", default=None)
    clothes: str | None = Field(
        description="Correctly describe the clothes the person is wearing on the image", default=None
    )


class Persona(ABSPersona, BaseModel):
    """
    A class representing a persona by parameters.

    Attributes:
        name (str): The name of the persona.
        gender (Gender): The gender of the persona.
        skintone (SkinTone): The skin tone of the persona.
        face (Face): Face describing the persona.
        body (Body): Body describing the persona.
        additional_parameters (dict): Additional parameters specific to the persona.
        random_person (bool): generate a random persona.

    Methods:
        prompt(): Returns the full prompt for the persona.

    """

    name: str | None = None
    gender: Gender | None = Field(description="Correctly assign gender of person on the image", default=None)
    age: PositiveInt = Field(description="Look carefully at the main person on the image and describe its age")
    skintone: SkinTone | None = Field(
        description="Correctly assign colour of skin of person on the image", default=None
    )
    race: str | None = Field(
        description="Look carefully at the main person on the image and describe its race", default=None
    )
    face: Face = Field(description="Look carefully at the main person on the image and describe its face")
    body: Body = Field(description="Look carefully at the main person on the image and describe its body")
    additional_parameters: str | None = Field(
        description="Look carefully at the main person on the image and describe parameters "
        "that are not basic, but are important in your opinion, what distinguishes this person."
        " You can leave this field blank.",
        default=None,
    )

    random_person: bool = Field(default=False, exclude=True)

    def __new__(cls, *args, **kwargs):
        cls._main_params = ["name", "gender", "skintone", "race"]
        cls._body_params = [p_name for p_name in Body.schema().get("properties") if p_name != "height"]
        cls._face_params = list(Face.schema().get("properties").keys())
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        random_person = kwargs.pop("random_person", False)

        if random_person:
            faker = Faker()
            kwargs["name"] = faker.name()
            gender = random.choice(list(Gender))
            kwargs["gender"] = gender
            kwargs["age"] = random.randint(13, 90)
            kwargs["skintone"] = random.choice(list(SkinTone))
            haircut = random.choice(list(HairCut))
            kwargs["face"] = Face(
                face_shape=random.choice(list(FaceShape)),
                beard=random.choice(list(Beard)) if gender == Gender.MALE else None,
                hair=haircut,
                hair_color=random.choice(list(HairColor)) if haircut != HairCut.BALD else None,
                eyes_color=random.choice(list(EyesColor)),
            )
            kwargs["body"] = Body(
                height=random.randint(140, 210),
                view=random.choice(list(ViewType)),
                body_shape=random.choice(list(BodyShape)),
            )
        # else:
        #     if not all((kwargs['name'] | args[0], kwargs['gender'] | args[1], kwargs['age'] | args[2])):
        #         raise ValueError("All mandatory parameters must be given (name, gender, age).")

        super().__init__(*args, **kwargs)

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
            prompt += f" with {self.skintone} skin tone, "
        if self.face.beard != Beard.CLEAN_SHAVED and self.gender == Gender.MALE:
            prompt += f"{self.face.beard} beard, "
        if self.face.hair_color:
            prompt += f"{self.face.hair_color} hair color, "
        if self.face.hair_style:
            prompt += f"{self.face.hair_style} hairstyle, "
        prompt += f"{self.face.face_shape} face shape, "
        prompt += f"{self.body.body_shape} body shape, "
        prompt += f"{self.body.height} cm tall, "
        if self.face.eyes_color:
            prompt += f"{self.face.eyes_color} eyes color, "
        if self.body.clothes:
            prompt += f"wearing {self.body.clothes}, "
        if self.additional_parameters:
            prompt += f". {self.additional_parameters}"

        return prompt

    @staticmethod
    def _most_common_params(counter: dict[str, Counter]) -> dict[str, str]:
        mcp = {}
        for p_name, p_counter in counter.items():
            if p_counter:
                most_common = p_counter.most_common(1)
                mcp[p_name] = most_common[0][0]
            else:
                mcp[p_name] = None

        return mcp

    @classmethod
    def _default_counter(cls):
        return {
            "main": {p_name: Counter() for p_name in cls._main_params},
            "body": {p_name: Counter() for p_name in cls._body_params},
            "face": {p_name: Counter() for p_name in cls._face_params},
        }

    @classmethod
    def mean(cls, people: Iterable["Persona"]) -> "Persona":
        counter = cls._default_counter()
        age_list = []
        height_list = []
        # count values
        for person in people:
            age_list.append(person.age)
            height_list.append(person.body.height)
            for param_name in cls._main_params:
                param = getattr(person, param_name, None)
                if param:
                    counter["main"][param_name].update((param,))

            for param_name in cls._body_params:
                param = getattr(person.body, param_name)
                if param:
                    counter["body"][param_name].update((param,))

            for param_name in cls._face_params:
                param = getattr(person.face, param_name)
                if param:
                    counter["face"][param_name].update((param,))

        # create mean person
        main_params = cls._most_common_params(counter["main"])
        body_params = cls._most_common_params(counter["body"])
        face_params = cls._most_common_params(counter["face"])
        mean_age = sum(age_list) // len(age_list)
        mean_height = sum(height_list) // len(height_list)
        mean_person = cls(
            age=mean_age, body=Body(**body_params, height=mean_height), face=Face(**face_params), **main_params
        )
        return mean_person

    def __str__(self):
        return self.prompt()

    def __repr__(self):
        return self.__str__()
