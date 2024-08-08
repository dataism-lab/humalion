from abc import ABC, abstractmethod
from enum import Enum


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

    Methods:
        prompt(): Returns the full prompt for the persona.

    """

    def __init__(
            self,
            name: str,
            gender: Gender,
            skintone: SkinTone,
            beard: Beard = Beard.CLEAN_SHAVED,
            hair_color: str = 'blonde',
            hair_style: str = '',
            face_shape: FaceShape = FaceShape.DIAMOND,
            body_shape: BodyShape = BodyShape.COLUMN,
            height: int = 180,
            closes: str = '',
            additional_parameters: dict | None = None,
    ):
        self.name = name
        self.gender = gender
        self.skintone = skintone
        self.beard = beard
        self.hair_color = hair_color
        self.hair_style = hair_style
        self.face_shape = face_shape
        self.body_shape = body_shape
        self.height = height
        self.closes = closes
        self.additional_parameters = additional_parameters

    def prompt(self) -> str:
        pass
