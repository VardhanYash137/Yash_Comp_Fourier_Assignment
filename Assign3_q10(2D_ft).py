#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(X,Y,Z,title):
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(title)
    plt.savefig("2D Gaussian")
    # Show plot
    plt.show()

# Define the function
def f1(x, y,key=0):
    if key==0:
        return np.exp(-x**2 - y**2)
    else:
        z_mat=np.zeros((len(x),len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                z_mat[i][j]=np.exp(-(x[i]**2+y[j]**2))
        return z_mat

def analytic_FT(u,v,key=0):
    return 0.5*np.exp(-(u**2 + v**2)/4) 
    

# Generate x and y values
N=int(128)
M=int(128)
x = np.linspace(-10,10, N)
y = np.linspace(-10, 10, M)
dx=x[1]-x[0]
dy=y[1]-y[0]
X, Y = np.meshgrid(x, y)
Z = f1(X, Y,key=0)
Z_mat=f1(x,y,key=1)
plot_3d(X,Y,Z,"Plot of e^(-(x^2+y^2))")

# Compute the numerical Fourier transform
Z_fft = (np.fft.fft2((Z_mat)))
k_x = 2*np.pi*(np.fft.fftfreq(N, dx))
k_y = 2*np.pi*(np.fft.fftfreq(M, dy))


Z_fft1=np.zeros((N,M))
for q in range(N):
    for r in range(M):
        phase_factor=(-1j)*((k_x[q]*(x[0]))+(k_y[r]*(y[0])))
        Z_fft1[q][r]=np.abs(((dx*dy)/(2*np.pi))*np.exp(phase_factor)*Z_fft[q][r])
        
# Compute the analytical Fourier transform
U, V = np.meshgrid(k_x,k_y)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 5), subplot_kw={'projection': '3d'})


axes[0].plot_surface( U,V, np.transpose(Z_fft1), cmap='plasma')
axes[0].set_title('Numerical Fourier Transform')
axes[0].set_xlabel('Frequency (k_x)')
axes[0].set_ylabel('Frequency (k_y)')
axes[0].set_zlabel('Magnitude')
plt.savefig("q10 Numerical FT")

# Analytical Fourier Transform
axes[1].plot_surface(U, V, analytic_FT(U,V), cmap='plasma')
axes[1].set_title('Analytical Fourier Transform')
axes[1].set_xlabel('Frequency (k_x)')
axes[1].set_ylabel('Frequency (k_y)')
axes[1].set_zlabel('Magnitude')
plt.savefig("q10 Analytic FT")

plt.tight_layout()
plt.show()




# In[ ]:




