#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return np.exp(-x**2)
    

def h(x):
    return np.exp(-4*(x**2))

    
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
x_min = -10
x_max = 10
n = 256
dx = (x_max-x_min)/(n-1)
x_arr=np.arange(x_min,x_max+dx,dx,float)

x_arr2=np.linspace(-x_max,x_max,1000,float)

g=np.vectorize(g)
h=np.vectorize(h)

g_arr=g(x_arr)
h_arr=h(x_arr)


# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first data
axes[0].scatter(x_arr, g_arr, c='red',label="sampled points",marker=".")
axes[0].plot(x_arr2, g(x_arr2), c='blue',label="curve")
axes[0].set_title('g(x)=e^(-x^2)')
axes[0].set_xlabel('x')
axes[0].grid()
axes[0].legend()


# Plot the second data
axes[1].scatter(x_arr, h_arr,c="purple",label="sampled points",marker=".")
axes[1].plot(x_arr2, h(x_arr2),c="green",label="curve")
axes[1].set_title('h(x)=e^(-4x^2)')
axes[1].set_xlabel('x')
axes[1].grid()
axes[1].legend()
# Adjust layout
plt.tight_layout()

plt.savefig("q12_g(x)_h(x)")
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
plt.scatter(x_arr,conv_arr,c="red",marker=".",label="numerical value")
plt.grid()
plt.plot(x_arr,conv_arr,label="analytic plot")
plt.legend()
plt.title("Convolution of f(x) and g(x)")
plt.ylabel("Conv(g,h)")
plt.xlabel("x")
plt.savefig("q12_conv(g,h)")
plt.show()


# In[ ]:




