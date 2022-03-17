
def function1(str):
    s = str.split(' ')
    print("The length of the last word is: ", len(s[-1]))
    print("The length of the sentence is: ", len(str))

def function2(list1, list2):
    merged_list = list1 + list2
    merged_list.sort()
    print("Merged and sorted list: ", merged_list)

def function3(list):
    list.sort()
    print("Second largest number in list: ", list[-2])

def function4(str):
    s = list(str)
    i = 0
    j = len(s)-1
    while i < j:
        if s[i].isalpha() and s[j].isalpha():
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1
        if not s[i].isalpha():
            i += 1
        if not s[j].isalpha():
            j -= 1
    reversed_string = ""
    print(reversed_string.join(s))
            
def function5(str):
    s = str.split(" ")
    s = s[::-1]
    reversed_string = " "
    print(reversed_string.join(s))

def function6(num1, num2):
    print(int(num1) + int(num2))

def function7(num1, num2):
    a = str(num1)
    b = str(num2)
    sum = bin(int(a, 2) + int(b, 2))
    print(sum[2:])

def function8(list1, list2=None):
    my_set = set(list1)
    if list2 is None:
        num_list = []
        occurrence_list = []
        for num in my_set:
            num_list.append(num)
            occurrence_list.append(list1.count(num))
        result = list(zip(num_list, occurrence_list))
        print(result)
        return
    intersection = my_set.intersection(list2)
    print(list(intersection))
    
def function9():
    string_from_user = input("Please type something: \n")

    d = dict()
    for char in string_from_user:
        if char in d:
            d[char] += 1
        else:
            d[char] = 1

    max_value = max(d.values())
    for key, value in d.items():
        if value == max_value:
            print(key)
        
    # frequencie = [(char, string_from_user.count(char)) for char in set(string_from_user)]
    # print(max(frequencie, key=lambda x: x[1])[0])

def function10(list):
    square_numbers = []
    for num in list:
        s = num**(0.5)
        if (s // 1) == s:
            square_numbers.append(num)
        else:
            continue
    print(square_numbers)

function1('Hello World!')
function2([0,1,2,8,-1, 9], [-11,9])
function3([-11, 1.2, 9.9,9])
function4('12311!hm')
function5('this is a sentence')
function6('28', '2')
function7(101, 10)
function8([5,55,1,12], [66,23,1,0,1,9])
function9()
function10([3, 7, 5, 55 ])