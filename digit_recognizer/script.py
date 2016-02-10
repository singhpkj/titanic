import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x= train.values[:, 1:]
y= train.values[:, 0]
test= test.values[:,0:]

print x.shape, y.shape

r= np.random.randint(1,5000, 9)
def plotRandomDigit(x, r, dim): # eg: from x choose 9 random digits and plot in 3x3 dimension 
    try:
	f, axarr= plt.subplots(dim,dim)
	for i  in range(dim):
    	    for j in range(dim):
        	a= np.reshape(x[r[i+j],:], (28,28))
        	axarr[i,j].imshow(a,  cmap='Greys')
	 	axarr[i,j].axis('off')
	#plt.show()
    except Exception as e:
        print str(e)
        print "Error in plotDigit"

#plotRandomDigit(x, r, 3)
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn import svm
"""
pca_range= [0.74, 0.76, 0.78, 0.80, 0.82, 0.84]
pca_score= []
for i in pca_range :
    pca = PCA(n_components=i,whiten=True)
    x1 = pca.fit_transform(x)
    clf= svm.SVC( )
    score = cross_validation.cross_val_score(clf, x1, y, cv=10)
    pca_score.append(score.mean() )


plt.plot(pca_range, pca_score)
plt.xlabel("values for n_component")
plt.ylabel("cross validation accuracy")
plt.show()
"""
pca = PCA(n_components=0.76,whiten=True)
x = pca.fit_transform(x)


#plotRandomDigit(x1, r, 3)
test = pca.transform(test)

clf= svm.SVC( )
#clf= svm.SVC(gamma=0.0001, C=100 )
clf.fit(x,y)
predictions= clf.predict(test)
#print predictions 
submission = pd.DataFrame( {"ImageId": range(1,len(predictions)+1), "Label": predictions} )
print submission.head(5)
submission.to_csv('output.csv', index=False, header=True)

#clf.predict(train.values[30,1:])
#train.values[30,0]
#cross validation
""" 
from sklearn import cross_validation
score = cross_validation.cross_val_score(clf, x, y, cv=3)
print score
"""
