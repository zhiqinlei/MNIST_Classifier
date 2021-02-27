import numpy as np 
import random 

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[21,22,23,24]])

b = np.array([2])
#b=2
# np.random.shuffle(a)

#print("a:\n", a)

for k in range (0,6,2):
    unit = a[k:k+2] 
    #print("unit:\n",unit)
    f = unit[:,:3]
    print("f",f)
    #print(type(f))
    print("b:\n", b)
    print("dot(f,b):\n", np.dot(f,int(b)))

