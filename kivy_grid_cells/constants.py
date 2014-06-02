class State(object):
    DEACTIVATED = 0
    ACTIVATED = 1

    @classmethod
    def get(cls, active):
        if active:
            return cls.ACTIVATED
        return cls.DEACTIVATED


class Colours(object):
    ACTIVATED = (1, 1, 1, 1)
    DEACTIVATED = (0.5, 0.5, 0.5, 1)
