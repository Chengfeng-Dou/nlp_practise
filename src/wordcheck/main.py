from src.wordcheck import Checker
from src.wordcheck import build_sm

if __name__ == '__main__':
    uInput = None
    checker = Checker(build_sm())
    print("Welcome to use Mr. Dou's spell checker")
    while True:
        print("Please input a word, type space to exit!")
        uInput = input()
        result = checker.check(uInput)
        if uInput == ' ':
            print('exit')
            break

        if len(result) == 0:
            print("Could you speak english?")
            continue

        if result[0] == uInput:
            print('The word is spelled correctly and well done!')
            continue

        print('Maybe these words are what you want?')
        print(result)
