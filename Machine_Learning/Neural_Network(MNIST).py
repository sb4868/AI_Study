import sys
sys.path.append('C:\Program Files\Python38\Lib\site-packages')
from keras.datasets import mnist
import numpy as np

(x_x_data, t_x_data), (x_t_data, t_t_data) = mnist.load_data()

x_data = np.loadtxt(r'C:\Users\USER\mnist\mnist_train.csv', delimiter=',', dtype = np.float32)
t_data = np.loadtxt(r'C:\Users\USER\mnist\mnist_test.csv', delimiter=',', dtype = np.float32)

import matplotlib.pyplot as plt

img = t_data[0][1:].reshape(28,28)
plt.imshow(img, cmap = 'gray')
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

class NerualNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.W2 = np.random.rand(self.input_nodes, self.hidden_nodes)
        self.b2 = np.random.rand(self.hidden_nodes)

        self.W3 = np.random.rand(self.hidden_nodes, self.output_nodes)
        self.b3 = np.random.rand(self.output_nodes)

        self.learning_rate = 1e-4

  def feed_forward(self):
      delta = 1e-7 

      z2 = np.dot(self.input_data, self.W2) + self.b2
      a2 = sigmoid(z2)

      z3 = np.dot(a2, self.W3) + self.b3
      a3 = sigmoid(z3)
      y = a3

      return -np.sum(self.target_data*np.log(y + delta) + (1-self.target_data)*np.log((1-y) + delta))

  def loss_val(self):
      delta = 1e-7 

      z2 = np.dot(self.input_data, self.W2) + self.b2
      a2 = sigmoid(z2)

      z3 = np.dot(a2, self.W3) + self.b3
      a3 = sigmoid(z3)
      y = a3

      return -np.sum(self.target_data*np.log(y + delta) + (1-self.target_data)*np.log((1-y) + delta))

  def train(self, training_data):

      self.target_data = np.zeros(self.output_nodes)
      self.target_data[int(training_data[0])] = 1
  
      self.input_data = (training_data[1:] / 255.0 * 1)
  
      f = lambda x : self.feed_forward()
  
      for step in range(10000):
          self.W2 -= self.learning_rate * numerical_derivative(f, self.W2)
          self.b2 -= self.learning_rate * numerical_derivative(f, self.b2)
          self.W3 -= self.learning_rate * numerical_derivative(f, self.W3)
          self.b3 -= self.learning_rate * numerical_derivative(f, self.b3)

  def predict(self, x_data):
      z2 = np.dot(x_data, self.W2) + self.b2
      a2 = sigmoid(z2)
  
      z3 = np.dot(a2, self.W3) + self.b3
      a3 = sigmoid(z3)

      predicted_num = np.argmax(a3)

      return predicted_num

  def accuracy(self, test_data):
      matched_list = []
      not_matched_list = []

    for i in range(len(test_data)):
        label = int(test_data[index, 0])
        data = (test_data[i, 1:] / 255.0)
        predicted_num = self.predict(data)
        if label == predicted_num:
           matched_list.append(index)
        else:
           not_matched_list.append(index)

      print("Current Accuracy =", 100*(len(matched_list)/(len(test_data))), "%")

      return matched_list, not_matched_list
  
  
# 학습 시작
a = NerualNetwork(784, 100, 10)

for step in range(30001):
    index = np.random.randint(0, len(t_data)-1)
    a.train(t_data[index])
    if step%400 == 0:
        print("step =",step, ", loss_val =", a.loss_val())
    
