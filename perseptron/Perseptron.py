'''
Created on 14 џэт. 2018 у.

@author: vkozhaev
'''
import numpy as np
class Perseptron(object):
    def __init__(self, eta=0.01,n_iter=10):
        self.eta=eta
        self.n_iter=n_iter
    
    def fit(self,X,y):
        self.w = np.zeros (1 + X.shape[1] )
        self.errors=[]
        
        for _ in range(self.n_iter):
            errors=0
            for xi,target in zip(X,y):
                update = self.eta * (target - self .predict (xi) )
                self.w_ [1:] += update * xi
                self . w_ [0] += update
                errors+=int(update!=0)
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:]+self.w_[0])

    def pred(self,X):
        return np.where(self.net_input(X)>=0,1,-1)  