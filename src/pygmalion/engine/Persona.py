from enum import Enum


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


class Persona:
    def __init__(
            self,
            name: str,
            gender: Gender,
            skintone: SkinTone,
            beard: Beard,
    ):
        self.name = name
        self.gender = gender
        self.skintone = skintone
        self.beard = beard
