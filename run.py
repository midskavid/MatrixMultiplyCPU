import os
import time
import numpy as np
import matplotlib.pyplot as plt


def process(fnm) :
    X = []
    Y = []

    with open(fnm,'r') as f:
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
    plt.title('Blocked')
    plt.savefig(fnm.split('.')[0]+'.png', dpi=300)
    plt.close()


L1 = [16,32,64]
L2 = [64, 80, 128]
L3 = [256, 512, 1024]


for l1 in L1: 
    for l2 in L2:
        for l3 in L3:
            cmd = 'gcc -O4 -mavx -mfma benchmark.c dgemm-blocked.c wall_time.c cmdLine.c -lcblas -lm -D BLOCK_SIZEL1='+str(l1)+' -D BLOCK_SIZEL2='+str(l2)+' -D BLOCK_SIZEL3='+str(l3)
            os.system(cmd)
            fnm = str(l1)+str(l2)+str(l3)+'.txt'
            os.system('./a.out > '+fnm)
            process(fnm)