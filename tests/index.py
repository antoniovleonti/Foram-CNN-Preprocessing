# Antonio Leonti
# 3.18.2020
# testing this for edm.py

import random

def main():
    l = [*range(20)]
    random.shuffle(l)

    i = [*range(20)]

    for x in l:
        print(x)

    for x in sorted(i, key = lambda _: l[_], reverse = True):
        print(l[x])

    # works perfectly

if __name__ == "__main__":
    main()
