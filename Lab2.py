#!/usr/bin/env python
# coding: utf-8

# In[404]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, ifft
from scipy.signal import convolve, filtfilt, butter, bartlett, freqz, cheby1, bessel, cheby2


# In[33]:


#Input signal
Input_1kHz_15kHz =[

+0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
-0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
+0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
+0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
-0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
-0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
+0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
-0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
-0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
]


# In[34]:


x_mag = np.array(Input_1kHz_15kHz)
x_len = np.linspace(0, (len(x_mag)), len(x_mag))


# In[35]:


#plot the frequency domain of input
#without stem
plt.figure(figsize=(10,2))
plt.plot(x_len, x_mag,".")
plt.title("Input signal in time domain")
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.grid()
plt.show()


# In[36]:


#convert into frequency domain
X = fft(x_mag)


# In[1153]:


#Input signal in frequency domain: -pi to pi

X1 = fftshift(X)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)
#plot X real, imag, mag, phase
fig = plt.gcf()

plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "mag in Db", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [dB]")
plt.title('Input signal amplitude: -pi to pi')
plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Magnitude")
plt.title('Input signal magnitude in frequency domain: -pi to pi')
plt.grid()

plt.subplot(2, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Input signal phase in frequency domain: -pi to pi')
plt.grid()
plt.show()


# FIR

# In[795]:


#Designing
h_Bartlett = np.bartlett(20)
H_Bartlett = fft(h_Bartlett, len(x_mag))
H_Bartlett_norm = H_Bartlett / abs(H_Bartlett).max()
h_Bartlett_norm = ifft(H_Bartlett_norm)
h_Bartlett_norm_final = h_Bartlett_norm[:20]
H_Bartlett_norm_final = fft(h_Bartlett_norm_final)
# w_test, H_test = freqz(h_Bartlett, 1, len(x_mag), whole = True)


# In[1084]:


plt.plot(h_Bartlett,".")
plt.title("Bartlett window")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.grid()


# In[796]:


# plt.plot(H_test)
# plt.plot(H_Bartlett)


# In[520]:


"""
#Bartlett in frequency domain
plt.plot(abs(fft(h_Bartlett)))
plt.grid()
"""


# In[521]:


"""
#Bartlett in frequency [Db] 
plt.plot(20*np.log10(abs(H_Bartlett)))
plt.grid()
"""


# In[1103]:


X1 = fftshift(H_Bartlett_norm)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)


# In[1106]:


fig = plt.gcf()

plt.subplot(2, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "Amplitude", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [dB]")
plt.title('Bartlett window (normalized) amplitude: -pi to pi')
plt.grid()

# plt.subplot(3, 2, 4)
# fig.set_size_inches(15,15)
# plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
# plt.legend()
# plt.xlabel("Frequency [rand/s]")
# plt.title('Bartlett window (normalized) magnitude in frequency domain: -pi to pi')
# plt.grid()

plt.subplot(2, 2, 4)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Bartlett window (normalized) phase in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[829]:


# Out_Bartlett_1 = ifft(np.multiply(fft(h_Bartlett_norm, len(x_mag)), X)).real
Out_Bartlett_2 = convolve(h_Bartlett_norm_final, x_mag)


fig = plt.gcf()

# plt.subplot(2, 2, 1)
# fig.set_size_inches(15,10)
# plt.plot(Out_Bartlett_1, ".",color = "purple")
# plt.xlabel("Samples")
# plt.ylabel("Amplitude [Db]")
# plt.title('Output using bartlett filter')
# plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(Out_Bartlett_2, ".",color = "red")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using bartlett filter in time domain')
plt.grid()


# In[831]:


#Input signal in frequency domain: -pi to pi
Out_Bartlett_2_freq = fft(Out_Bartlett_2)
X1 = fftshift(Out_Bartlett_2_freq)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)
#plot X real, imag, mag, phase
fig = plt.gcf()

plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "mag in Db", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [Db]")
plt.title('Output signal after using bartlett window in frequency doamin: -pi to pi')
plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Magnitude")
plt.title('Output signal after using bartlett window in frequency domain: -pi to pi')
plt.grid()

plt.subplot(2, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Output signal after using bartlett window in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[1154]:


h_Hamming = np.hamming(20)
H_Hamming = fft(h_Hamming, len(x_mag))
H_Hamming_norm = H_Hamming / abs(H_Hamming).max()
h_Hamming_norm = ifft(H_Hamming_norm)
h_Hamming_norm_final = h_Hamming_norm[:20]
H_Hamming_norm_final = fft(h_Hamming_norm_final)


# In[1155]:


plt.plot(h_Hamming_norm_final,".")
plt.title("Hamming window normalized")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.grid()


# In[1113]:


X1 = fftshift(H_Hamming_norm)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)


# In[1114]:


fig = plt.gcf()

plt.subplot(3, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "Amplitude", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [dB]")
plt.title('Hamming window (normalized) amplitude: -pi to pi')
plt.grid()

# plt.subplot(3, 2, 4)
# fig.set_size_inches(15,15)
# plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
# plt.legend()
# plt.xlabel("Frequency [rand/s]")
# plt.title('Hamming window (normalized) magnitude in frequency domain: -pi to pi')
# plt.grid()

plt.subplot(3, 2, 4)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Hamming window (normalized) phase in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[804]:


# Out_Hamming_1 = ifft(np.multiply(fft(h_Hamming_norm, len(x_mag)), X)).real
Out_Hamming_2 = convolve(h_Hamming_norm_final, x_mag)


fig = plt.gcf()

# plt.subplot(2, 2, 1)
# fig.set_size_inches(15,10)
# plt.plot(Out_Hamming_1, ".", color = "purple")
# plt.xlabel("Samples")
# plt.ylabel("Magnitude")
# plt.title('Output using hamming window')
# plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(Out_Hamming_2,".", color = "red")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using hamming window in time domain')
plt.grid()


# In[1156]:


#Input signal in frequency domain: -pi to pi
Out_Hamming_2_freq = fft(Out_Hamming_2)
X1 = fftshift(Out_Hamming_2_freq)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)
#plot X real, imag, mag, phase
fig = plt.gcf()

plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "mag in Db", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [dB]")
plt.title('Output signal after using hamming window in frequency doamin: -pi to pi')
plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Magnitude")
plt.title('Output signal after using hamming window in frequency domain: -pi to pi')
plt.grid()

plt.subplot(2, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Output signal after using hamming window in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[1110]:


h_Blackman = np.blackman(20)
H_Blackman = fft(h_Blackman, len(x_mag))
H_Blackman_norm = H_Blackman / abs(H_Blackman).max()
h_Blackman_norm = ifft(H_Blackman_norm)
h_Blackman_norm_final = h_Blackman_norm[:20]
H_Blackman_norm_final = fft(h_Blackman_norm_final)


# In[1090]:


plt.plot(h_Blackman_norm_final,".")
plt.title("Blackman window normalized")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.grid()


# In[1111]:


X1 = fftshift(H_Blackman_norm)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)


# In[1112]:


fig = plt.gcf()

plt.subplot(3, 2, 3)
fig.set_size_inches(15,15)
plt.plot(w1, 20*np.log10(abs(X1)), label = "Amplitude", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [dB]")
plt.title('Blackman window (normalized) amplitude: -pi to pi')
plt.grid()

# plt.subplot(3, 2, 4)
# fig.set_size_inches(15,15)
# plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
# plt.legend()
# plt.xlabel("Frequency [rand/s]")
# plt.title('Blackman window (normalized) magnitude in frequency domain: -pi to pi')
# plt.grid()

plt.subplot(3, 2, 4)
fig.set_size_inches(15,15)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Blackman window (normalized) phase in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[1123]:


# Out_Blackman_1 = ifft(np.multiply(fft(h_Blackman_norm, len(x_mag)), X)).real
Out_Blackman_2 = convolve(h_Blackman_norm_final, x_mag)


fig = plt.gcf()

# plt.subplot(2, 2, 1)
# fig.set_size_inches(15,10)
# plt.plot(Out_Blackman_1, ".", color = "purple")
# plt.xlabel("Samples")
# plt.ylabel("Magnitude")
# plt.title('Output using bartlett window')
# plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(Out_Blackman_2,".", color = "red")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using blackman window in time domain')
plt.grid()


# In[810]:


#Input signal in frequency domain: -pi to pi
Out_Blackman_2_freq = fft(Out_Blackman_2)
X1 = fftshift(Out_Blackman_2_freq)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)
#plot X real, imag, mag, phase
fig = plt.gcf()

plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "mag in Db", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [Db]")
plt.title('Output signal after using blackman window in frequency doamin: -pi to pi')
plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Magnitude")
plt.title('Output signal after using blackman window in frequency domain: -pi to pi')
plt.grid()

plt.subplot(2, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Output signal after using blackman window in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[1127]:


# Out_Bartlett_1 = ifft(np.multiply(fft(h_Bartlett_norm, len(x_mag)), X)).real
Out_Bartlett_2 = convolve(h_Bartlett_norm_final, x_mag)


fig = plt.gcf()

plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(x_mag, ".",color = "red")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Input in time domain')
plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(Out_Bartlett_2, ".",color = "blue")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using bartlett filter in time domain')
plt.grid()

plt.subplot(2, 2, 3)
fig.set_size_inches(15,10)
plt.plot(Out_Hamming_2, ".",color = "green")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using hamming filter in time domain')
plt.grid()

plt.subplot(2, 2, 4)
fig.set_size_inches(15,10)
plt.plot(Out_Blackman_2, ".",color = "yellow")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using blackman filter in time domain')
plt.grid()


# In[811]:


# plt.plot(F3)
# plt.plot(F2)
# plt.plot(F1, color = "pink")


# IRF

# In[1091]:


fig = plt.gcf()

plt.subplot(2,1,1)
fig.set_size_inches(15,10)
Fs = len(x_mag)
X_f = abs(fft(x_mag))
l = len(x_mag)
fr = (Fs/2)*np.linspace(0,1,int(l/2))
xl_m_o = (2/l)*abs(X_f[0:np.size(fr)])
# plt.plot(20* np.log10(xl_m_o))
plt.plot(xl_m_o)
plt.title("Input affecting frequency")
plt.ylabel("Magnitude")
plt.xlabel("Frequency samples")
plt.grid()


# In[1132]:


o = 20
fc = 40 #critical frequency
wc = (2*fc)/Fs
b, a = butter(o, wc , btype = "low", analog = False)
w, H_butter = freqz(b, a, len(x_mag), whole = True)
x_filtered_butter = filtfilt(b, a, x_mag)
# h_butter = ifft(H_butter).real


# In[1133]:


plt.plot(ifft(H_butter),".")
plt.title("Butterworth impulse response")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.grid()


# In[1134]:


X1 = fftshift(H_butter)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)


# In[1135]:


fig = plt.gcf()

plt.subplot(3, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "Amplitude", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [dB]")
plt.title('Butterworth filter amplitude: -pi to pi')
plt.grid()

# plt.subplot(3, 2, 4)
# fig.set_size_inches(15,15)
# plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
# plt.legend()
# plt.xlabel("Frequency [rand/s]")
# plt.title('Butterworth filter magnitude in frequency domain: -pi to pi')
# plt.grid()

plt.subplot(3, 2, 4)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Butterworth filter phase in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[1136]:


Out_Butterworth_1 = ifft(np.multiply(H_butter, X)).real
Out_Blackman_2 = convolve(h_butter, x_mag)

fig = plt.gcf()

plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(x_filtered_butter, ".", color = "purple")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using butterworth filter')
plt.grid()


# In[1157]:


Out_Butterworth_2_freq = fft(x_filtered_butter)
X1 = fftshift(Out_Butterworth_2_freq)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)
fig = plt.gcf()

plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "mag in Db", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [dB]")
plt.title('Output signal after using butterworth filter in frequency domain: -pi to pi')
plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Magnitude")
plt.title('Output signal after using butterworth filter in frequency domain: -pi to pi')
plt.grid()

plt.subplot(2, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Output signal after using butterworth filter in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[1138]:


o = 20
fc = 40 #critical frequency
wc = (2*fc)/Fs
b, a = bessel(o, wc , btype = "low", analog = False)
w, H_bessel = freqz(b, a, len(x_mag), whole = True)

x_filtered_bessel = filtfilt(b, a, x_mag)

# plt.plot(abs(fft(x_mag)))
# plt.plot(abs(fft(x_filtered)))


# In[1139]:


plt.plot(ifft(H_bessel),".")
plt.title("Bessel impulse response")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.grid()


# In[1140]:


X1 = fftshift(H_bessel)


# In[1141]:


fig = plt.gcf()

plt.subplot(3, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "Amplitude", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [dB]")
plt.title('Bessel filter amplitude: -pi to pi')
plt.grid()

# plt.subplot(3, 2, 4)
# fig.set_size_inches(15,15)
# plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
# plt.legend()
# plt.xlabel("Frequency [rand/s]")
# plt.title('Bessel filter magnitude in frequency domain: -pi to pi')
# plt.grid()

plt.subplot(3, 2, 4)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Bessel filter phase in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[1142]:


fig = plt.gcf()

plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(x_filtered_bessel, ".", color = "purple")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using bessel filter')
plt.grid()


# In[1143]:


Out_Bessel_2_freq = fft(x_filtered)
X1 = fftshift(Out_Bessel_2_freq)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)
fig = plt.gcf()

plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "mag in Db", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [Db]")
plt.title('Output signal after using bessel filter in frequency domain: -pi to pi')
plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Magnitude")
plt.title('Output signal after using bessel filter in frequency domain: -pi to pi')
plt.grid()

plt.subplot(2, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Output signal after using bessel filter in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[1144]:


o = 20
fc = 40 #critical frequency
wc = (2*fc)/Fs
b, a = cheby1(o, 1 ,wc , btype = "low", analog = False)
w, H_chebyshev = freqz(b, a, len(x_mag), whole = True)

x_filtered_chebyshev = filtfilt(b, a, x_mag)

# # plt.plot(abs(fft(x_mag)))
# plt.plot(abs(fft(x_mag)))
# plt.plot(abs(fft(x_filtered)))


# In[1145]:


plt.plot(ifft(H_chebyshev),".")
plt.title("Chebyshev impulse response")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.grid()


# In[1146]:


X1 = fftshift(H_chebyshev)


# In[1147]:


fig = plt.gcf()

plt.subplot(3, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "Amplitude", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [dB]")
plt.title('Chebyshev filter amplitude: -pi to pi')
plt.grid()

# plt.subplot(3, 2, 4)
# fig.set_size_inches(15,15)
# plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
# plt.legend()
# plt.xlabel("Frequency [rand/s]")
# plt.title('Chebyshev filter magnitude in frequency domain: -pi to pi')
# plt.grid()

plt.subplot(3, 2, 4)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Chebyshev filter phase in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[1148]:


fig = plt.gcf()

plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(x_filtered_chebyshev, ".", color = "purple")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using chebyshev filter')
plt.grid()


# In[1149]:


Out_Bessel_2_freq = fft(x_filtered)
X1 = fftshift(Out_Bessel_2_freq)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)
fig = plt.gcf()

plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "mag in Db", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [Db]")
plt.title('Output signal after using bessel filter in frequency doamin: -pi to pi')
plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Magnitude")
plt.title('Output signal after using bessel filter in frequency domain: -pi to pi')
plt.grid()

plt.subplot(2, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Output signal after using bessel filter in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[1152]:


fig = plt.gcf()
plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(x_mag, ".",color = "red")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Input in time domain')
plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(x_filtered_chebyshev, ".",color = "blue")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using chebyshev filter in time domain')
plt.grid()

plt.subplot(2, 2, 3)
fig.set_size_inches(15,10)
plt.plot(x_filtered_butter, ".",color = "green")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using butterworth filter in time domain')
plt.grid()

plt.subplot(2, 2, 4)
fig.set_size_inches(15,10)
plt.plot(x_filtered_bessel, ".",color = "yellow")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using bessel filter in time domain')
plt.grid()


# Implement a highpass filter to remove the low-frequency component in labwork 1

# In[859]:


#Using FIR - Hamming window
h_Hamming_high = np.hamming(100)
H_Hamming_high = fft(h_Hamming, len(x_mag))
H_Hamming_high_norm = H_Hamming / abs(H_Hamming).max()
h_Hamming_high_norm = ifft(H_Hamming_norm)
h_Hamming_high_norm_final = h_Hamming_norm[:20]
H_Hamming__high_norm_final = fft(h_Hamming_norm_final)
plt.plot(abs(H_Hamming_high))


# In[1158]:


#Using butterworth filter
o = 20
fc = 40 #critical frequency
wc = (2*fc)/Fs
b, a = butter(o, wc , btype = "highpass", analog = False)
w, H_butter_high = freqz(b, a, len(x_mag), whole = True)
x_filtered_butter_high = filtfilt(b, a, x_mag)
# h_butter = ifft(H_butter).real


# In[1159]:


X1 = fftshift(H_butter_high)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)


# In[1160]:


fig = plt.gcf()

plt.subplot(3, 2, 3)
fig.set_size_inches(15,15)
plt.plot(w1, 20*np.log10(abs(X1)), label = "Amplitude", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [Db]")
plt.title('Butterworth filter amplitude: -pi to pi')
plt.grid()

plt.subplot(3, 2, 4)
fig.set_size_inches(15,15)
plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Butterworth filter magnitude in frequency domain: -pi to pi')
plt.grid()

plt.subplot(3, 2, 5)
fig.set_size_inches(15,15)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Butterworth filter phase in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[1163]:


Out_Butterworth_2_freq = fft(x_filtered_butter_high)
X1 = fftshift(Out_Butterworth_2_freq)
w1 = np.arange(-np.pi, np.pi, 2*np.pi/X1.size)
fig = plt.gcf()

plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(w1, 20*np.log10(abs(X1)), label = "mag in dB", color = "purple")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Amplitude [Db]")
plt.title('Output signal after using butterworth filter in frequency doamin: -pi to pi')
plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(w1, np.abs(X1), label = "mag", color = "yellow")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.ylabel("Magnitude")
plt.title('Output signal after using butterworth filter in frequency domain: -pi to pi')
plt.grid()

plt.subplot(2, 2, 3)
fig.set_size_inches(15,10)
plt.plot(w1, np.angle(X1), label = "phase", color = "blue")
plt.legend()
plt.xlabel("Frequency [rand/s]")
plt.title('Output signal after using butterworth filter in frequency domain: -pi to pi')
plt.grid()
plt.show()


# In[876]:


Out_Butterworth_1 = ifft(np.multiply(H_butter_high, X)).real
Out_Blackman_2 = convolve(h_butter, x_mag)

fig = plt.gcf()
plt.subplot(2, 2, 1)
fig.set_size_inches(15,10)
plt.plot(x_filtered, ".", color = "purple")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using butterworth filter')
plt.grid()

plt.subplot(2, 2, 2)
fig.set_size_inches(15,10)
plt.plot(x_filtered, color = "purple")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.title('Output using butterworth filter (smoothed out)')
plt.grid()


# In[882]:


from scipy.io import wavfile


# In[885]:


cd Desktop


# In[1050]:


samplerate, data = wavfile.read("Audio1_1.wav")


# In[1164]:


plt.title("All samples")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.plot(data)
plt.grid()


# In[1165]:


#adding noise
mean = 0
std = 0.01
num_samples = len(data)
np.random.seed(100) #using a seed

white_noise_samples = np.random.normal(mean, std, size=num_samples)
data_noise = data + white_noise_samples
# ECG_add_noise_freq = fft(ECG_add_noise)


# In[1053]:


plt.plot(abs(fft(white_noise_samples)))
plt.grid()


# In[1170]:


plt.plot(data_noise, label = "With noise")
plt.plot(data, label = "Original")
plt.legend()
plt.title("Audio with noise generated")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()


# In[1055]:


Fs = samplerate
tstep = 1/Fs
N = len(data)
t = np.linspace(0, (N-1)*tstep, N)
fstep = Fs/N
f = np.linspace(0, (N-1)*fstep, N)


# In[1019]:


# X_data = scipy.fft.rfft(data)


# In[1056]:


X_data = fft(data)


# In[1172]:


fig = plt.gcf()

plt.subplot(2,1,1)
fig.set_size_inches(15,10)
Fs = samplerate
X_f = abs(fft(data))
l = len(data)
fr = (Fs/2)*np.linspace(0,1,int(l/2))
xl_m_o = X_f[0:np.size(fr)]
# plt.plot(20* np.log10(xl_m_o))
plt.plot(fr,xl_m_o)
plt.title("Original affecting frequency")
plt.ylabel("Magnitude")
plt.xlabel("Frequency [Hz]")
plt.grid()

plt.subplot(2,1,2)
Fs = samplerate
X_f_noise = abs(fft(data_noise))
l = len(data)
fr = (Fs/2)*np.linspace(0,1,int(l/2))
xl_m = X_f_noise[0:np.size(fr)]
# plt.plot(20* np.log10(xl_m_o))
plt.plot(fr,xl_m)
plt.title("Noise frequency")
plt.ylabel("Magnitude")
plt.xlabel("Frequency [Hz]")
plt.grid()


# In[1174]:


fig = plt.gcf()

plt.subplot(2,1,1)
fig.set_size_inches(15,10)
Fs = samplerate
X_f = abs(fft(data))
l = len(data)
fr = (Fs/2)*np.linspace(0,1,int(l/2))
xl_m_o = X_f[0:np.size(fr)]
# plt.plot(20* np.log10(xl_m_o))
plt.plot(fr,20*np.log10(xl_m_o))
plt.title("Original affecting frequency")
plt.ylabel("Magnitude [dB]")
plt.xlabel("Frequency [Hz]")
plt.grid()

plt.subplot(2,1,2)
Fs = samplerate
X_f_noise = abs(fft(data_noise))
l = len(data)
fr = (Fs/2)*np.linspace(0,1,int(l/2))
xl_m = X_f_noise[0:np.size(fr)]
# plt.plot(20* np.log10(xl_m_o))
plt.plot(fr,20*np.log10(xl_m))
plt.title("Noise frequency")
plt.ylabel("Magnitude [dB]")
plt.xlabel("Frequency [Hz]")
plt.grid()


# In[1059]:


X_white_noise = fft(white_noise_samples)


# In[1060]:


fig = plt.gcf()

plt.subplot(2,1,1)
fig.set_size_inches(15,10)
Fs = samplerate
X_f_whitenoise = abs(fft(white_noise_samples))
l = len(white_noise_samples)
fr = (Fs/2)*np.linspace(0,1,int(l/2))
xl_m_o_white_noise = (2/l)*abs(X_f_whitenoise[0:np.size(fr)])
# plt.plot(20* np.log10(xl_m_o))
plt.plot(fr,xl_m_o_white_noise)
plt.title("Original affecting frequency")
plt.ylabel("Magnitude")
plt.xlabel("Frequency [Hz]")
plt.grid()


# In[1181]:


#create a lowpass
Fs = samplerate
o = 5
fc = 3000 #critical frequency
wc = 2*fc/Fs
b, a = butter(o, wc , btype = "low", analog = False)
data_filtered = filtfilt(b, a, data_noise)
plt.plot(data_filtered, label = "Filtered")
# plt.plot(data, label = "Original", color = "yellow")
plt.legend()
plt.title("Audio after filtering")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()


# In[1176]:


fig = plt.gcf()

plt.subplot(2,1,1)
fig.set_size_inches(15,10)
Fs = samplerate
X_f_filtered = abs(fft(data_filtered))
l = len(data)
fr = (Fs/2)*np.linspace(0,1,int(l/2))
xl_m_filter = X_f_filtered[0:np.size(fr)]
# plt.plot(20* np.log10(xl_m_o))
plt.plot(fr,20*np.log10(xl_m_filter))
plt.title("After filtered frequency")
plt.ylabel("Magnitude [dB]")
plt.xlabel("Frequency [Hz]")
plt.grid()

plt.subplot(2,1,2)
Fs = samplerate
X_f_noise = abs(fft(data_noise))
l = len(data)
fr = (Fs/2)*np.linspace(0,1,int(l/2))
xl_m = X_f_noise[0:np.size(fr)]
# plt.plot(20* np.log10(xl_m_o))
plt.plot(fr,20*np.log10(xl_m))
plt.title("Noise frequency")
plt.ylabel("Magnitude [dB]")
plt.xlabel("Frequency [Hz]")
plt.grid()


# In[1177]:


fig = plt.gcf()

plt.subplot(2,1,1)
fig.set_size_inches(15,10)
Fs = samplerate
X_f_filtered = abs(fft(data_filtered))
l = len(data)
fr = (Fs/2)*np.linspace(0,1,int(l/2))
xl_m_filter = X_f_filtered[0:np.size(fr)]
# plt.plot(20* np.log10(xl_m_o))
plt.plot(fr,xl_m_filter)
plt.title("After filtered frequency")
plt.ylabel("Magnitude")
plt.xlabel("Frequency [Hz]")
plt.grid()

plt.subplot(2,1,2)
Fs = samplerate
X_f_noise = abs(fft(data_noise))
l = len(data)
fr = (Fs/2)*np.linspace(0,1,int(l/2))
xl_m = X_f_noise[0:np.size(fr)]
# plt.plot(20* np.log10(xl_m_o))
plt.plot(fr,xl_m)
plt.title("Noise frequency")
plt.ylabel("Magnitude")
plt.xlabel("Frequency [Hz]")
plt.grid()


# In[1073]:


wavfile.write('noise.wav', samplerate, data_noise)

