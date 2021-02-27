# 1. Build a function that given two strings "s1" and "s2", creates a new string by appending "s2" in the middle of "s1"

def middle_string(s1, s2):

    middle_s1 = len(s1) // 2
    new_string = s1[:middle_s1] + s2 + s1[middle_s1:]

    print('New string:', new_string)


middle_string('something', 'great')
print('-----------'*5)


# 2. Build a function that, given a tring, reorders its characters so all the uppercase letters come first,
# and the lowercase letter come last

def reorder_uppercase(word):

    uppercase = []
    lowercase = []

    for i in word:
        if i.isalnum():
            if i.isupper():
                uppercase.append(i)
            else:
                lowercase.append(i)

    final_string = ''.join(uppercase + lowercase)

    print('Reordered string:', final_string)

reorder_uppercase('Hello World')


# 3. Build a function that, given a list of values, returns a dictionary with the number of ocurrencies of each
# value in the list

