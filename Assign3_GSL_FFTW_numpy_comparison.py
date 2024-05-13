#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt

def calculate_FT(data_points,dx,x_min):
    dft_arr=np.fft.fft(data_points,norm="ortho") #dft f(k_q) 
    k_arr=2*np.pi*np.fft.fftfreq(len(data_points),dx)
    fk_arr=np.zeros(len(k_arr))  #initialising to save value of fourier transform of f(x)
    for i in range(len(k_arr)): 
        fk_arr[i]=np.abs(dx*np.sqrt(len(data_points)/(2*np.pi))*np.exp(-(1j)*k_arr[i]*x_min)*(dft_arr[i])) #x_min=0
    return k_arr,fk_arr

def sinc(x):
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

def read_data_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Assuming data is whitespace-separated
            values = line.strip().split()
            # Convert values to float if necessary
            data.append([float(value) for value in values])
    return data


fftw_arr = read_data_file("fftw_sinc_data.txt")

#Parameters of Program
x_min = -20
x_max = 20
n = 128
dx = (x_max-x_min)/(n-1)
x_arr=np.arange(x_min,x_max+dx,dx,float)
sinc=np.vectorize(sinc)
k_arr,fk_arr=calculate_FT(sinc(x_arr),dx,x_min)

plt.plot(np.sort(k_arr),sort_with_order(k_arr,fftw_arr),label="fftw")
plt.plot(np.sort(k_arr),sort_with_order(k_arr,fk_arr),"--",label="python")
plt.xlabel("k")
plt.ylabel("f'(k)")
plt.legend()
plt.grid()
plt.title("FT of sinc using FFTW and python")
plt.savefig("q2_fftw")
plt.show()




# In[20]:


gsl_arr=read_data_file("gsl_sinc_data.txt")
plt.plot(np.sort(k_arr),sort_with_order(k_arr,fftw_arr),label="fftw")
plt.plot(np.sort(k_arr),sort_with_order(k_arr,fk_arr),"--",label="python")
plt.plot(np.sort(k_arr),sort_with_order(k_arr,gsl_arr),"--",label="GSL")
plt.xlabel("k")
plt.ylabel("f'(k)")
plt.legend()
plt.grid()
plt.title("FT of sinc using FFTW, GSL and python")
plt.savefig("q3_gsl")
plt.show()



# In[21]:


def gauss_analytic(x):
    return (1/np.sqrt(2))*np.exp((-x**2)/4)

gauss_analytic=np.vectorize(gauss_analytic)
k_arr,fk_arr=calculate_FT(np.exp(-1*np.square(x_arr)),dx,x_min)
gauss_arr=read_data_file("fftw_gauss_data.txt")
plt.plot(np.sort(k_arr),sort_with_order(k_arr,gauss_arr),label="fftw")
plt.plot(np.sort(k_arr),sort_with_order(k_arr,fk_arr),"--",label="python")
plt.plot(np.sort(k_arr),sort_with_order(k_arr,gauss_analytic(k_arr)),"--",label="analytic")
plt.xlabel("k")
plt.ylabel("f'(k)")
plt.legend()
plt.grid()
plt.title("FT of sinc using FFTW, GSL and python")
plt.savefig("q4_fftw")
plt.show()


# In[ ]:




