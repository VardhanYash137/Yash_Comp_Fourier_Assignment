#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Coded by: Yash Vardhan

import numpy as np
import matplotlib.pyplot as plt

def sync(x):
    if x==0:
        return 1
    else:
        return (np.sin(x)/x)
    
def sort_with_order(arr1, arr2):
    # Create pairs of elements from arr1 and arr2
    pairs = [(arr1[i], arr2[i]) for i in range(len(arr1))]
    
    # Sort pairs based on arr1
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    
    # Extract sorted elements from arr2
    sorted_arr2 = [pair[1] for pair in sorted_pairs]
    
    return sorted_arr2

def rect_func(k):
    if k>-1 and k<1:
        return np.sqrt(np.pi/2)
    else:
        return 0

#Parameters of Program
x_min = -20
x_max = 20
n = 128
dx = (x_max-x_min)/(n-1)
x_arr=np.arange(x_min,x_max+dx,dx,float)

sync=np.vectorize(sync) #converts a scalar function to vector function

#Plotting sync function
plt.plot(x_arr,sync(x_arr),label="sinc(x)")
plt.scatter(x_arr,sync(x_arr),c="red",marker=".",label="Sampled Points")
plt.xlabel("x")
plt.ylabel("sinc(x)")
plt.grid()
plt.title("Plot of Sinc(x)")
#plt.savefig("Plot of Sinc(x)")
plt.show()

dft_arr=np.fft.fft(sync(x_arr),norm="ortho") #dft f(k_q)
k_arr=2*np.pi*np.fft.fftfreq(n,dx)           #array of k_q's

fk_arr=np.zeros(len(k_arr))  #initialising to save value of fourier transform of f(x)
for i in range(len(k_arr)):
    fk_arr[i]=np.abs(dx*np.sqrt(n/(2*np.pi))*np.exp(-(1j)*k_arr[i]*x_min)*(dft_arr[i]))
    
plt.scatter(k_arr,fk_arr,c="red",marker=".",label="numerical")
plt.plot(np.sort(k_arr),sort_with_order(k_arr,fk_arr),c="red") #it just make ploting ordered according to k-scale
rect_func=np.vectorize(rect_func)
plt.plot(np.sort(k_arr),sort_with_order(k_arr,rect_func((k_arr))),label="Analytic")
plt.grid()
plt.xlabel("k")
plt.ylabel("f'(k)")
plt.title("FT of sinc function")
#plt.savefig("FT of sinc function")
plt.legend()
plt.show()

