#%%

"""
1D list's DFT algorithm
"""

import numpy as np 
import math
import cmath
from matplotlib import pyplot as plt

x = np.random.rand(150)
print("Size of Given Datas: ",len(x))

def generate_dft_matrix(N):
    dft = np.zeros((N,N))
    for n in range(N):
        for k in range(N):
            dft[n,k] = np.exp(-2j*np.pi*k*n/N)
    return dft

def dft(x):
    N = len(x)
    dft_matrix = generate_dft_matrix(N)
    return np.dot(dft_matrix,x)

X = dft(x)
print("Size of Discrete Fourier Transform of Given Datas: ",len(X))

print("x:",x)
print("DFT(x): ",X)

plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1)
plt.plot(X)
plt.title("Fourier Transform of Discrete Data")

#plt.title("x")
plt.subplot(1, 2, 2)
plt.plot(x)
plt.title("Given Discrete Data")
plt.style.use("seaborn")
plt.show()


#%%

"""

1D List's DFT algorithm(Modified)

"""

x = [1, 2, 3 ,4 ,5]

def Discrete_Fourier_Transform(N):
    N = len(x)
    print("N",N)
    dft_matrix = np.zeros((N,N))
    for i in range(N):
        for k in range(N):
            dft_matrix[i,k] = np.exp(-2j*np.pi*k*i/N)
    print(dft_matrix)
    dot_product = np.dot(dft_matrix,x)
    print(dot_product)
    return dot_product

X = Discrete_Fourier_Transform(x)

print("x:",x)
print("DFT(x): ",X)

plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1)
plt.plot(X)
plt.title("Fourier Transform of Discrete Data")

plt.subplot(1, 2, 2)
plt.plot(x)
plt.title("Given Discrete Data")
plt.style.use("seaborn")
plt.show()


#%%

"""
DFT algorithm for 2D array
"""

a = [ [1, -1, 1, -1, 5, 4, 3, 2],
      [1, -1, 1, -1, 5, 4, 3, 2] ]

a = np.asarray(a)
print(type(a))

def dft(input_img):
    rows = input_img.shape[0]
    cols = input_img.shape[1]
    output_img = np.zeros((rows,cols),complex)
    for m in range(0,rows):
        for n in range(0,cols):
            for x in range(0,rows):
                for y in range(0,cols):
                    output_img[m][n] += input_img[x][y] * np.exp(-1j*2*math.pi*(m*x/rows+n*y/cols))
    return output_img
dft(a)


# %%

"""
Nayuki's DFT algorithm

"""
import cmath
def compute_dft_complex(input):
	n = len(input)
	output = []
	for k in range(n):  # For each output element
		s = complex(0)
		for t in range(n):  # For each input element
			angle = 2j * cmath.pi * t * k / n
			s += input[t] * cmath.exp(-angle)
		output.append(s)
	return output

a = [1,2,3,4]
compute_dft_complex(a)

# %%

