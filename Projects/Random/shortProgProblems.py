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
    final_list1 = []
    final_list2 = []
    for i in range(len(list1)):
        for j in range(1, len(list1)):
            l1 = list1[i:j]
            if len(l1) > 0:
                final_list1.append(l1)

    for k in range(len(list2)):
        for l in range(1, len(list2)):
            l2 = list2[k:l]
            if len(l2) > 0:
                final_list2.append(l2)

    length = []
    for m in final_list1:
        for n in final_list2:
            if m == n and len(m) > len(length):
                length = m

    print('Longest common subarray:', length)


common_subarray(main_list, sub_list)
print('-------------' * 5)


# -----------------------------------------------------

def common_subarray_improved(list1, list2):
    last = [0 for _ in range(len(list2) + 1)]
    max_length = 0

    for i in range(1, len(list1) + 1):
        current = [0 for _ in range(len(list1) + 1)]

        for j in range(1, len(list2) + 1):
            if list1[i - 1] == list2[j - 1]:
                current[j] = 1 + last[j - 1]
                max_length = max(max_length, current[j])

        last = current
    print('Longest common subarray:', max_length)


common_subarray_improved(main_list, sub_list)
print('-------------' * 5)

# 9. Write a function that finds the length of the shortest array that contains both input arrays as subarrays.

A = [5, 6, 5, 2, 7, 5, 6, 7, 5, 5, 7, 3, 4, 8, 6]
B = [2, 7, 5, 6, 7, 5]


def shortest_common_subarray(list1, list2):
    result = 1000000
    n = len(list1)
    m = len(list2)

    # Traverse main_array element
    for i in range(0, n - m + 1):

        # Pick only those subarray of main array whose first element match with the first element of second array
        if list1[i] == list2[0]:

            # initialize starting of both subarrays
            j = 0
            index = i
            for index in range(i, n):
                if list1[index] == list2[j]:
                    j = j + 1

                # if we found all elements of the second array
                if j == m:
                    break

            # update minimum length sub_array
            if j == m and result > index - i + 1:
                if index == n:
                    result = index - i
                else:
                    result = index - i + 1

    # return minimum length subarray
    print('Length of shortest array:', result)


shortest_common_subarray(A, B)
print('-------------' * 5)

# 10. Write a function that, given an array, determines if you can partition it in two separate subarrays such that
# the sum of elements in both subarrays is the same.

sum_array = [2, 3, 6, 5, 4, 1, 2, 3, 4, 5, 10, 3, 4]


def sum_split_array(arr):
    n = len(arr)

    for i in range(0, n - 1):
        left_arr = arr[:i + 1]
        right_arr = arr[i + 1:]

        sum_right = sum(right_arr)
        sum_left = sum(left_arr)

        if sum_right == sum_left:
            print('Subarray I:', left_arr, '| Sum:', sum_left)
            print('Subarray II:', right_arr, '| Sum:', sum_right)


sum_split_array(sum_array)
print('-------------' * 5)

# 11. Write a function that, given an array, divides it into two subarrays, such as the absolute difference
# between their sums is minimum.


sub_array = [2, 3, 6, 5, 4, 1, 2, 8, 5, 20, 3, 4, 5, 14, 3, 4]


def sum_split_array(arr):
    n = len(arr)

    left_arr = [0]
    right_arr = [0]
    abs_diff = 10 ** 10

    for i in range(0, n):
        left_arr_int = arr[:i]
        right_arr_int = arr[i:]

        sum_right = sum(right_arr_int)
        sum_left = sum(left_arr_int)

        abs_diff_int = abs(sum_right - sum_left)

        if abs_diff_int < abs_diff:
            abs_diff, left_arr, right_arr = abs_diff_int, left_arr_int, right_arr_int

    print('Subarray I:', left_arr, '| Sum:', sum(left_arr))
    print('Subarray II:', right_arr, '| Sum:', sum(right_arr))
    print('Minimum absolute difference:', abs_diff)


sum_split_array(sub_array)
print('-------------' * 5)


# --------------------------------------------------------------------------

def short_sum_split_array(arr):
    n = len(arr)
    abs_diff = 10 ** 10
    left_arr = []
    right_arr = []

    for i in range(0, n):
        if abs(sum(arr[:i]) - sum(arr[i:])) < abs_diff:
            abs_diff, left_arr, right_arr = abs(sum(arr[:i]) - sum(arr[i:])), arr[:i], arr[i:]

    print('Subarray I:', left_arr, '| Sum:', sum(left_arr))
    print('Subarray II:', right_arr, '| Sum:', sum(right_arr))
    print('Minimum absolute difference:', abs_diff)


short_sum_split_array(sub_array)
