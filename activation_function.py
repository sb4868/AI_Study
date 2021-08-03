import numpy as np
import matplotlib.pylab as plt

def step_function(x):
  return np.array(x > 0, dtype=np.int)
  
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximun(0, x)

def softmax(x):
  if x.ndim == 2:
    x = x.T
    x = x -np.max(x, axis = 0)
    y = np.exp(x) / np.sum(np.exp(x), axis = 0)
    return y.T
    
'''    
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1)
plt.plot(x, y2, 'k--')
plt.ylim(-0.1, 1.1)
plt.show()
'''

x = np.array([1.0, 0.5])
w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])

print(w1.shape)
print(x.shape)
print(b1.shape)
a1 = np.dot(x, w1) + b1
z1 = sigmoid(a1)