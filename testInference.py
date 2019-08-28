from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

Err=np.load('Error_array.npy')
Lkl=np.load('Likelihood_array.npy')
plt.subplot(2,1,1)
plt.plot(Err)
#plt.draw()
plt.subplot(212)
plt.plot(-Lkl)
plt.show()
