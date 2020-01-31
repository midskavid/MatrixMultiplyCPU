import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

with open(sys.argv[1],'r') as f:
    for line in f:
        words = line.strip().split()
        
        if len(words)>0 and words[0] == 'Size:' :
            X.append(int(words[1]))
            Y.append(float(words[-1]))

X = np.array(X, dtype=np.int32)
Y = np.array(Y, dtype=np.float32)


plt.plot(X,Y, 'or--')
plt.grid()
plt.xlabel('Matrix size')
plt.ylabel('GFlops/s')
plt.title(sys.argv[2])
plt.savefig(sys.argv[1].split('.')[0]+'.png', dpi=300)
