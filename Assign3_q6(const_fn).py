#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Coded by: Yash Vardhan

import numpy as np
import matplotlib.pyplot as plt

def const_func(x,c=1):
    return c*np.ones(len(x))
    
def sort_with_order(arr1, arr2):
    # Create pairs of elements from arr1 and arr2
    pairs = [(arr1[i], arr2[i]) for i in range(len(arr1))]
    
    # Sort pairs based on arr1
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    
    # Extract sorted elements from arr2
    sorted_arr2 = [pair[1] for pair in sorted_pairs]
    
    return sorted_arr2


#Parameters of Program
x_min = -20
x_max = 20
n = 128
dx = (x_max-x_min)/(n-1)
x_arr=np.arange(x_min,x_max+dx,dx,float)

#Plotting sync function
plt.plot(x_arr,const_func(x_arr))
plt.xlabel("x")
plt.ylabel("const(x)")
plt.grid()
plt.title("Constant Function")
plt.savefig("Constant Fn")
plt.show()

dft_arr=np.fft.fft(const_func(x_arr),norm="ortho") #dft f(k_q)
k_arr=2*np.pi*np.fft.fftfreq(n,dx)           #array of k_q's

fk_arr=np.zeros(len(k_arr))  #initialising to save value of fourier transform of f(x)
for i in range(len(k_arr)):
    fk_arr[i]=np.abs(dx*np.sqrt(n/(2*np.pi))*np.exp(-(1j)*k_arr[i]*x_min)*(dft_arr[i]))
    
plt.scatter(k_arr,fk_arr,c="red",marker=".",label="numerical")
plt.plot(np.sort(k_arr),sort_with_order(k_arr,fk_arr)) #it just make ploting ordered according to k-scale
plt.grid()
plt.xlabel("k")
plt.ylabel("f'(k)")
plt.title("FT of const function")
plt.legend()
plt.savefig("FT of const function")
plt.show()


# In[ ]:




