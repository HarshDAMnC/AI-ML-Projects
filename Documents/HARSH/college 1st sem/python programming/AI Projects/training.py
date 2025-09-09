import numpy as np

np.random.seed(42)

x=np.array([[1,1],[1,0],[0,0]])
y=np.array([[1],[1],[0]])

learningrate=0.5

w=np.random.randn(2,1)
b=np.random.randn(1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))

for epoch in range(10000):
    z=np.dot(x,w)+b
    a=sigmoid(z)
    loss=np.mean((a-y)**2)
    
    dz=2*(a-y)*sigmoid_deriv(z)
    dw=np.dot(x.T,dz)/len(x)
    db=np.mean(dz)
    
    w-=learningrate*dw
    b-=learningrate*db
    
    if(epoch%100==0):
        print("courrent loss value=",loss)
        
print("\nfinal predictions:")
print("output :",sigmoid(np.dot(x,w)+b))

