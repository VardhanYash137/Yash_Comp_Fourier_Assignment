import numpy as np
import matplotlib.pyplot as plt

   
def sort_with_order(arr1, arr2):
    # Create pairs of elements from arr1 and arr2
    pairs = [(arr1[i], arr2[i]) for i in range(len(arr1))]
    
    # Sort pairs based on arr1
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    
    # Extract sorted elements from arr2
    sorted_arr2 = [pair[1] for pair in sorted_pairs]
    
    return sorted_arr2

def calculate_FT(data_points,dx=1,x_min=0):
    dft_arr=np.fft.fft(data_points,norm="ortho") #dft f(k_q) 
    k_arr=2*np.pi*np.fft.fftfreq(len(data_points),dx)
    fk_arr=np.zeros(len(k_arr))  #initialising to save value of fourier transform of f(x)
    for i in range(len(k_arr)): 
        fk_arr[i]=np.abs(dx*np.sqrt(len(data_points)/(2*np.pi))*np.exp(-(1j)*k_arr[i]*x_min)*(dft_arr[i])) #x_min=0
    return fk_arr

    

# Initialize an empty list to store data points
data_points = []

# Open the file in read mode
with open('noise.txt', 'r') as file:
    # Iterate over each line in the file
    for data in file:
        # Append the data points to the list
        data_points.append(float(data))

n=len(data_points)
print(n)
plt.title("Data Points")
plt.plot(range(1,n+1,1),data_points)
plt.ylabel("Noise")
plt.xlabel("count")
plt.grid()
#plt.savefig("q13_noise")
plt.show()

dft_data=np.abs(np.fft.fft(data_points,norm="ortho")) #dft f(k_q)
k_arr=2*np.pi*np.fft.fftfreq(n,1)  
plt.title("DFT")
plt.plot(np.sort(k_arr),sort_with_order(k_arr,dft_data))
plt.ylabel("f'(k)")
plt.xlabel("k")
plt.grid()
#plt.savefig("DFT_noise")
plt.show()

PS2=np.square(np.abs(calculate_FT(data_points)))
plt.title("Power Spectrum")
plt.plot(np.sort(k_arr),sort_with_order(k_arr,PS2))
plt.ylabel("PS(k)")
plt.xlabel("k")
plt.grid()
plt.savefig("PS_noise")
plt.show()
   
num_bins = 10
bin_size = len(PS2) // num_bins
binned_power_spectrum = [bin_size*np.mean(PS2[i*bin_size:(i+1)*bin_size]) for i in range(num_bins)]


plt.figure(figsize=(10, 5))
plt.plot(binned_power_spectrum, marker='o')
plt.title('Binned Power Spectrum')
plt.xlabel('k bins')
plt.ylabel('Power')
plt.grid()
plt.savefig("Binned PS")
plt.show()
