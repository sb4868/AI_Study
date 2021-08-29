import numpy as np

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
  
# 클래스 선언
class LogicGate:

  def __init__(self, gate_name, x_data, t_data):
      self.name = gate_name

      # 입력 데이터, 정답 데이터 초기화
      self.x_data = x_data.reshape(4,2)
      self.t_data = t_data.reshape(4,1)

      # 2층 hidden layer
      self.W2 = np.random.rand(2, 6)
      self.b2 = np.random.rand(6)

      # 3층 output layer
      self.W3 = np.random.rand(6, 1)
      self.b3 = np.random.rand(1)

      self.learning_rate = 1e-2

      print(self.name, " object is created")

  def feed_forward(self):
      delta = 1e-7 

      z2 = np.dot(self.x_data, self.W2) + self.b2
      a2 = sigmoid(z2)

      z3 = np.dot(a2, self.W3) + self.b3
      a3 = sigmoid(z3)
      y = a3

      return -np.sum(self.t_data*np.log(y + delta) + (1-self.t_data)*np.log((1-y) + delta))

  def loss_val(self):
      delta = 1e-7 

      z2 = np.dot(self.x_data, self.W2) + self.b2
      a2 = sigmoid(z2)

      z3 = np.dot(a2, self.W3) + self.b3
      a3 = sigmoid(z3)
      y = a3

      return -np.sum(self.t_data*np.log(y + delta) + (1-self.t_data)*np.log((1-y) + delta))

  def train(self):

      f = lambda x : self.feed_forward()

      print("Initial loss value =", self.loss_val())

      for step in range(10001):
          self.W2 -= self.learning_rate * numerical_derivative(f, self.W2)
          self.b2 -= self.learning_rate * numerical_derivative(f, self.b2)
          self.W3 -= self.learning_rate * numerical_derivative(f, self.W3)
          self.b3 -= self.learning_rate * numerical_derivative(f, self.b3)
          if (step % 400) == 0:
              print("step =", step, ", loss vlaue = ", self.loss_val())

  def predict(self, x_data):

      z2 = np.dot(x_data, self.W2) + self.b2
      a2 = sigmoid(z2)

      z3 = np.dot(a2, self.W3) + self.b3
      a3 = sigmoid(z3)
      y = a3

      if (y > 0.5).all():
          result = 1
      else:
          result = 0

      return y, result
    

# 결과 확인
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_data = np.array([0, 1, 1, 0])
a = LogicGate('aa', x_data, t_data)
a.train()
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data in test_data:
    print(a.predict(data))
