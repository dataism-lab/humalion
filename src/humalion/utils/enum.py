from enum import EnumMeta, Enum


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class StrEnum(str, Enum, metaclass=MetaEnum):
    pass