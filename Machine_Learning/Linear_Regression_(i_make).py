import numpy as np
import random as rd

t_input = np.array([1, 2, 3, 4, 5, 6])
t_output = np.array([10, 20, 30, 40, 50, 60])

weight = rd.randint(1, 1000)
bias = rd.randint(1, 1000)
print(weight, bias)


def mean_square(func):    
    return np.sqrt(np.sum((t_output - func)**2))
    
def diff_w(weight, bias):
    s_value = 0.000001
    func = weight * t_input + bias
    d_func = (weight - s_value) * t_input + bias
    return (mean_square(func)-mean_square(d_func))/s_value

def diff_b(weight, bias):
    s_value = 0.00001
    func = weight * t_input + bias
    d_func = weight * t_input + (bias - s_value)
    return (mean_square(func)-mean_square(d_func))/s_value

learning_rate = 0.01

for i in range(100000):
    weight -= learning_rate*diff_w(weight, bias)
    bias -= learning_rate*diff_b(weight, bias)
    if i % 500 == 0:
        print(weight, bias)
    
    
print(weight, bias)
def test(value):
    return weight * value + bias
