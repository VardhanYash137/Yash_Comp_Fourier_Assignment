#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt

def f(x):
    if -1<x and x<1:
        return 1
    else:
        return 0

    
def sort_with_order(arr1, arr2):
    # Create pairs of elements from arr1 and arr2
    pairs = [(arr1[i], arr2[i]) for i in range(len(arr1))]
    
    # Sort pairs based on arr1
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    
    # Extract sorted elements from arr2
    sorted_arr2 = [pair[1] for pair in sorted_pairs]
    
    return sorted_arr2

def analytic_plot(x):
    return np.sqrt(np.pi/5)*np.exp((-x**2)*(4/5))


#Parameters of Program
x_min = -5
x_max = 5
n = 128
dx = (x_max-x_min)/(n-1)
x_arr=np.arange(x_min,x_max+dx,dx,float)

x_arr2=np.linspace(-x_max,x_max,1000,float)

f=np.vectorize(f)

g_arr=f(x_arr)
h_arr=f(x_arr)



# Plot the first data
plt.scatter(x_arr, g_arr, c='red',label="sampled points",marker=".")
plt.plot(x_arr2, f(x_arr2), c='blue',label="curve")
plt.ylabel('f(x)')
plt.xlabel('x')
plt.grid()
plt.legend()
# Adjust layout
plt.tight_layout()
plt.savefig("q11_f(x)")
# Show the plots
plt.show()


g_pad_arr=np.pad(g_arr, (0, len(g_arr)), mode='constant')
h_pad_arr=np.pad(h_arr, (0, len(h_arr)), mode='constant')


dft_g_arr=np.fft.fft(g_pad_arr,norm="ortho")
dft_h_arr=np.fft.fft(h_pad_arr,norm="ortho")
 
gh_arr=np.multiply(dft_g_arr,dft_h_arr)

ift_arr=np.fft.ifft(gh_arr,norm="ortho") 
conv_arr=(dx*np.sqrt(2*n))*np.abs(ift_arr)


conv_arr=conv_arr[int(n/2):int(-n/2)]
plt.grid()
plt.plot(x_arr,conv_arr)
plt.title("Convolution of f(x) with itself")
plt.ylabel("f*f(x)")
plt.xlabel("x")
plt.savefig("q11_Convolution_f(x)")
plt.show()


# In[ ]:




