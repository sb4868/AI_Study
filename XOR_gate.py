# encoding: utf-8

from and_gate import AND
from or_gate import OR
from Nand_gate import NAND

def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y
  
if __name__ == '__main__':
  for ar in [(0,0), (1,0), (0,1), (1,1)]:
    y = XOR(ar[0], ar[1])
    print(str(ar) + " -> " + str(y))