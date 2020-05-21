class State(object):

    def __init__(self, value=None, is_final=False):
        self._value = value
        self._is_final = is_final
        self._next = None

    @property
    def is_final(self):
        return self._is_final

    @is_final.setter
    def is_final(self, is_final: bool):
        self._is_final = is_final

    @property
    def value(self):
        return self._value

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, state):
        self._next = state
