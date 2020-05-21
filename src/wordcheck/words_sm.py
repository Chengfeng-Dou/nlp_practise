from src.wordcheck import State


def find_next(char, candidate: list):
    if candidate is None:
        return None

    for state in candidate:
        if state.value == char:
            return state
    return None


class WordsSM:
    def __init__(self):
        self._root = State()
        words_start = [State(chr(ord('a') + i)) for i in range(26)]
        self._root.next = words_start

    def add_word(self, word: str):
        if len(word) == 0:
            pass

        word = word.lower()
        cur_state = self._root

        for char in word:
            next_state = find_next(char, cur_state.next)
            if next_state is None:
                next_state = State(char)

                if cur_state.next is None:
                    cur_state.next = [next_state]
                else:
                    cur_state.next.append(next_state)

            cur_state = next_state

        cur_state.is_final = True

    @property
    def root(self):
        return self._root


def build_sm():
    w_sm = WordsSM()
    with open("words.txt") as f:
        for line in f:
            w_sm.add_word(line.strip())
    return w_sm
