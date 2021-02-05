# short programming problems by Santiago: https://twitter.com/svpino/status/1354048200601198593?s=20

# 1. Write a function that reverses an array in place.
a = [1, 2, 3, 4, 5, 6, 7, 8]


def reverse(l):
    l = l[::-1]
    return l


print(reverse(a))
print('-------------' * 5)

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
print('-------------' * 5)

# 3. Write a function that finds the duplicate number in an unsorted array containing every number from 1 to 100.

c = [7, 5, 6, 2, 1, 3, 4, 5, 8, 10, 9]  # 5 is duplicate


def find_duplicate_number(lst):
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]

    for k in range(len(lst) - 1):
        if lst[k] == lst[k + 1]:
            print('{} is duplicated'.format(lst[k]))


find_duplicate_number(c)
print('-------------' * 5)

# 4. Write a function that removes every duplicate value in an array.

d = [7, 5, 6, 2, 1, 3, 4, 5, 6, 2, 8, 10, 9]  # 5, 2, and 6 are duplicates


def remove_duplicate_number(lst):
    res = []
    for i in lst:
        if i not in res:
            res.append(i)
    print('Final list: ', res)


remove_duplicate_number(d)
print('-------------' * 5)

# 5. Write a function that finds the largest and smallest number in an unsorted array.

e = [7, 5, 6, 2, 1, 3, 4, 5, 6, 14, 20, 0, 2, 8, 10, 9]


def find_small_large(lst):
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]

    print('Smallest number in the list is:', lst[0])
    print('Largest number in the list is:', lst[-1])
    print(lst)


find_small_large(e)
print('-------------' * 5)

# ANOTHER WAY: 5. Write a function that finds the largest and smallest number in an unsorted array.

e = [7, 5, 6, 2, 1, 3, 4, 5, 6, 14, 20, 0, 2, 8, 10, 9]


def find_small_large(lst):
    min_number = lst[0]
    max_number = lst[0]
    for i in range(len(lst)):
        if lst[i] < min_number:
            min_number = lst[i]
        elif lst[i] > max_number:
            max_number = lst[i]
    print('Smallest number in the list is:', min_number)
    print('Largest number in the list is:', max_number)
    print(lst)


find_small_large(e)
print('-------------' * 5)

# 6. Write a function that finds a subarray whose sum is equal to a given value.

f = [7, 5, 6, 2, 1, 3, 4, 5, 6, 11, 12, 14, 20, 13, 15, 16, 17, 0, 2, 8, 10, 9]


def sum_array(lst, n):
    for i in range(len(lst)):
        for j in range(1 + i, len(lst)):
            if sum(lst[i:j]) == n:
                print('Subarray whose sum is equal to {}:'.format(n), lst[i:j])


sum_array(f, 24)
print('-------------' * 5)

# 7. Write a function that finds the contiguous subarray of a given size with the largest sum.

g = [7, 5, 6, 2, 1, 3, 4, 5, 6, 11, 12, 14, 20, 13, 15, 16, 17, 0, 2, 8, 10, 9]


def largest_sum(lst, size):
    list_chunk = lst[0:size]
    largest_sum = sum(list_chunk)
    for i in range(len(lst) - size):
        if sum(lst[i:i + size]) > largest_sum:
            list_chunk = lst[i:i + size]
            largest_sum = sum(lst[i:i + size])

    print('Subarray of size {}:'.format(size), list_chunk)
    print('Sum:', largest_sum)


largest_sum(g, 7)
print('-------------' * 5)

# 8. Write a function that, given two arrays, finds the longest common subarray present in both of them.

main_list = [1, 2, 3, 4, 5, 6, 7, 8]
sub_list = [1, 2, 3, 4, 7, 9]


def common_subarray(list1, list2):
    length_list = 0
    final_common_list = []
    k = 0

    for i in range(len(list1)):
        placeholder_list = list1[k:i]
        print(placeholder_list.intersection(list2))
        # if placeholder_list.intersection(list2) == placeholder_list:
        print('True')#and len(placeholder_list)>length_list:
        final_common_list = placeholder_list
        length_list = len(placeholder_list)

        print(length_list)
        print(placeholder_list)

common_subarray(main_list, sub_list)