# 1. Build a function that given two strings "s1" and "s2", creates a new string by appending "s2" in the middle of "s1"

def middle_string(s1, s2):

    middle_s1 = len(s1) // 2
    new_string = s1[:middle_s1] + s2 + s1[middle_s1:]

    print('New string:', new_string)


middle_string('something', 'great')
# New string: somegreatthing
