# coding: utf-8
import numpy as np

def OR(x1, x2):
  x = np.array([x1, x2])
  w = np.array([1, 1])
  b = 0
  y = np.sum(w*x) + b
  
  if y >= 1:
    return 1
  else:
    return 0

if __name__ == '__main__':
  for ar in [(0,0), (1,0), (0,1), (1,1)]:
    y = OR(ar[0], ar[1])
    print(str(ar) + " -> " + str(y))
