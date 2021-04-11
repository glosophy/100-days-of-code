# Write a  program to check a list is empty or not.
empty = []
not_empty = [1, 2, 3]


def check_empty(list1):
    if len(list1) == 0:
        print('List is empty.')
    else:
        print('List is not empty.')


check_empty(empty)
check_empty(not_empty)

# Write a program that simulates the rolling of a die.
from random import randrange


def roll_die():
    x = int(input('How many times do you want to roll a die?'))

    for i in range(x):
        die = randrange(1, 7)
        print('Roll #{}: {}'.format(i + 1, die))


roll_die()


# Write a Python program to implement pow(x, n).

def power(x, n):
    """Calculates a number elevated to a certain power"""
    p = x ** n
    print(p)


power(2, 9)
