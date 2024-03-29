import numpy as np

x_data = np.array([[0,0],[1,0],[0,1],[1,1]])
t_data = np.array([0, 1, 1, 0]).reshape(4,1)

W = np.random.rand(2, 1)
b = np.random.rand(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cost(x, t):
    delta = 1e-7 
    z = np.dot(x, W) + b
    y = sigmoid(z)
    return -np.sum(t*np.log(y + delta) + (1-t)*np.log((1-y) + delta))

def numerical_derivative(f, x):
    delta_x = 1e-7
    grad = np.zeros_like(x)     
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
    
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)
        x[idx] = tmp_val - delta_x
        fx2 = f(x)
        
        grad[idx] = (fx1 - fx2) / (2 * delta_x)
        x[idx] = tmp_val
        it.iternext()
        
    return grad

    
f = lambda x : cost(x_data, t_data)

learning_rate = 1e-2

for step in range(80001):
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)
    if (step%10000 == 0):
        print("step =", step, ", error value :", cost(x_data, t_data), ", W =",W, ", b =",b)
        

def test(x):
    z = np.dot(x, W) + b
    y = sigmoid(z)
    if y > 0.5:
        result = 'pass'
    else:
        result = 'fail'
    return y, result
