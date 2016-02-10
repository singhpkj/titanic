import matplotlib.pyplot as plt
import numpy as np


def plotDigit(a):
    try:
        a= np.reshape(a, (28,28))
        plt.imshow(a)
        plt.show()
    except Exception as e:
        print str(e)
        print "Error in plotDigit"

def M(x, p, q):
    try:
        m= 0
        a= np.reshape(x, (28,28))
        for i in range(28):
            for j in range(28):
                m+= i**p * j**q * a[i,j]
        return m
    except Exception as e:
        print str(e)
        print "Error in calculating M"


def centroidsList(x):
    try:
        c= []
        for i in range(x.shape[0]):
            x_bar, y_bar= M(x[i,:], 1, 0)/M(x[i,:], 0, 0),  M(x[i,:], 0, 1)/M(x[i,:], 0, 0)
            c.append( [x_bar, y_bar] )
        return c
    except Exception as e:
        print str(e)
        print "Error in centroidsList"


def translate_img(x, c):
    try:
        x1= np.ndarray(x.shape, dtype=int)
        for i in range(x.shape[0]):
            a= np.reshape(x[i,:], (28,28))
            a1= np.reshape(x1[i,:], (28,28))
            shift=[14-c[i][0], 14-c[i][1]  ]
            for j in range(28):
                for k in range(28):
                    if j+shift[0] >= 28:
                        j1= j+shift[0] -28
                    else:
                        j1= j+shift[0]

                    if k+shift[1] >= 28:
                        k1= k+shift[1] -28
                    else:
                        k1= k+shift[1]
                    a1[j1,k1]= a[j, k]
            x1[i,:]= np.ravel(a1)


        return x1

    except Exception as e:
        print str(e)
        print "Error in translate_img"

"""
def testC(c):
    sums=0
    for i in c:
        if i[0] >=16 or i[1] >=16:
            sums+=1
    print sums
"""
c= centroidsList(x)
#print c[:1000]
x= translate_img(x, c)
#c1= centroidsList(x1)
#print c1[:100]
#x= np.reshape(train.values[30,1:], (28,28))
#print Means(x,0,1)

