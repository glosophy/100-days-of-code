# short programming problems by Santiago: https://twitter.com/svpino/status/1354048200601198593?s=20

# 1. Write a function that reverses an array in place.
a = [1, 2, 3, 4, 5, 6, 7, 8]


def reverse(l):
    l = l[::-1]
    return l


print(reverse(a))

# 2. Write a function that finds the missing number in an unsorted array containing every one of
# the other 99 numbers ranging from 1 to 100.

b = [7, 5, 6, 2, 1, 3]  # 4 is missing


def find_missing_number(lst):
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]

    for k in range(len(lst) - 1):
        sub = lst[k + 1] - lst[k]
        if sub == 2:
            print('Missing value between {} and {}'.format(lst[k], lst[k + 1]))


find_missing_number(b)
# Missing value between 3 and 5
