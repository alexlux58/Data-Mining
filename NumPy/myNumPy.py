import random
import numpy as np


a = np.array([[1,2,3],[2.2, 3.2, 4.1]], dtype='float64')

print(a)
print(a.ndim)

# m x n
print(a.shape)

# matrix data type
print(a.dtype)

# elements in matrix
print(a.size)

# size of data type
print(a.itemsize)

# size * itemsize = nbytes
print(a.nbytes)

# get an element
print(a[1,2])

# print the row
print(a[0, :])

# print the column
print(a[:, 2])

# change an element
a[0,0] = 20

print(a)

# init zero matrix
z = np.zeros((3,3))
print(z)

# ones matrix
o = np.ones((3,2))
print(o)

# any other number matrix
any_matrix = np.full((2,2), 100)
print(any_matrix)

# random matrix values between 0 and 1
rand_matrix = np.random.rand(4,2)
print(rand_matrix)

# random integer matrix
rand_int_matrix = np.random.randint(7, size=(4,4,2))
print(rand_int_matrix)

# identity matrix
id = np.identity(5)
print(id)

print()

# creating a custom matrix
output = np.ones((5,5))
z[1,1] = 9
output[1:4, 1:4] = z
print(output)

# copy matrix 
cpy = output.copy()
print(cpy)

# can perform any math operation on the matrix
# ex:
a += 2 * 3 - 1 / 1.5
c = np.sin(a)
print(a)
print('\n', c)

# can perform linear algebra operations on matricies
deter = np.linalg.det(id)
print(deter)

# can also perform statistic operations
min_value = np.min(a, axis=0) # printed on row bases
print(min_value)

# can use reshape to change the size of the array
# use genfromtxt('data.txt', delimiter=',') to get the matrix from a file
# look into boolean masking and advanced indexing