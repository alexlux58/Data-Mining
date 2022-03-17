
'''
Week 1: Lab Assignment
'''

######### (100 points) ###########

'''
   1) write a function that gets a sentence (as a str) and outputs the length of the last word and the count of the characters in the sentence. 
   
   e.g.: input_sentence = 'Hello World!' --> output = [ 5 , 12 ]       
   e.g.: input_sentence = 'This is a sentence.' --> output = [ 8 , 19]
   

'''
def function1(str):
    s = str.split(' ')
    print("The length of the last word is: ", len(s[-1]))
    print("The length of the sentence is: ", len(str))

'''
    2)  write a function that gets two lists of numbers and merges them and outputs the sorted merged list.
   
   e.g.: input_list1 = [0,1,2,8,-1, 9] , input_list2 = [-11,9]--> output = [-11, -1, 0, 1, 2, 8, 9, 9]
   
'''
def function2(list1, list2):
    merged_list = list1 + list2
    merged_list.sort()
    print("Merged and sorted list: ", merged_list)

'''
    3)  write a function that get a list of numbers and outputs the second largest number in it.
   
   e.g.: input_list = [0,1,2,8,-1] --> output = 2
   e.g.: input_list = [-11, 1.2, 9.9,9] --> output = 9
'''
def function3(list):
    list.sort()
    print("Second largest number in list: ", list[-2])


'''
    4)  write a function that get a string as an input and outputs the reverse of that (only letters!)
   
   e.g.: input_str = 'hel-lo,wo.rld' --> output = 'dlr-ow,ol.leh'
   e.g.: input_str = '12311!hm' --> output = '12311!mh'
'''
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


'''
    5)  write a function that gets a sentence (as a str) and outputs the reverse of that.
   
   e.g.: input_sentence = 'Hello World' --> output = 'World Hello'
   e.g.: input_sentence = 'this is a sentence' --> output = 'sentence a is this'
'''
def function5(str):
    s = str.split(" ")
    s = s[::-1]
    reversed_string = " "
    print(reversed_string.join(s))



'''
    6)  write a function that gets two numbers as strings and outputs the sum of them. 
   
   e.g.: str1 = '96' , sr2 = '21'--> output = 117
   e.g.: str1 = '28' , sr2 = '2'--> output = 30
'''
def function6(num1, num2):
    print(int(num1) + int(num2))

'''
    7)  write a function that gets two binary numbers from the user and outputs the sum of them in binary. 
   
   e.g.: input1 = 101 , input2 = 10 --> output = 111
   e.g.: input1 = 11 , input2 = 1 --> output = 100
'''
def function7(num1, num2):
    a = str(num1)
    b = str(num2)
    sum = bin(int(a, 2) + int(b, 2))
    print(sum[2:])


'''
    8)  write a function:
    
        5-1) thats gets a list of numbers and counts the occurrences of all items in it. 
              e.g.: list1 = [9,9,1,0,1,9] --> output = [(9,3) , (1,2), (0,1)]
              
        5-2) that gets two lists and outputs their common elements. 
             e.g.: list1 = [66,23,1,0,1,9] , list2 = [5,55,1,12] --> output = [1]         

'''
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



'''
    9)  write a function that gets a string from the user and returns the most frequent character in it. 
   
   e.g.: str1 = 'hello world'--> output = 'l'
   e.g.: str1 = 'cs_comp478'--> output = 'c'
'''
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



'''
    10) write a function that gets a list of numbers and returns the compnents of the list that are perfect square numbers. 
   
   e.g.: input1 = [1, 5, 8, 9] --> output = [1,9]
   e.g.: input1 = [3, 7, 5, 55 ]--> output = []

'''

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