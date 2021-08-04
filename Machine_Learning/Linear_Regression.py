import numpy as np
import random as rd

x_data = np.array([9, 14, 21, 27, 32, 37]).reshape(6,1)
t_data = np.array([75, 81, 86, 90, 88, 92]).reshape(6,1)

W = np.random.rand(1,1)
b = np.random.rand(1)

def cost(x, t):
    y = np.dot(x, W) + b
    return np.sum((t - y)**2) / (len(x))

def numerical_derivative(f, x):
    delta_x = 1e-5
    grad = np.zeros_like(x)     
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
    
        x[idx] = tmp_val + delta_x
        fx1 = f(x)
        x[idx] = tmp_val - delta_x
        fx2 = f(x)
        
        grad[idx] = (fx1 - fx2) / (2 * delta_x)
        x[idx] = tmp_val
        it.iternext()
        
    return grad

def test(x):
    y = np.dot(x,W) + b
    return y

f = lambda x : cost(x_data, t_data)

def learn(x_data, t_data, W, b, learning_rate):
    for step in range(8001):
        W -= learning_rate * numerical_derivative(f, W)
        b -= learning_rate * numerical_derivative(f, b)
        if (step%1000 == 0):
            print("step =", step, ", error value :", cost(x_data, t_data), ", W =",W, ", b =",b)
            
loaded_data = np.loadtxt('./data_01.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[ :, 0:-1]
t_data = loaded_data[ :, [-1]]

W = np.random.rand(3,1)
b = np.random.rand(1)

learn(x_data, t_data, W, b, 1e-5)
