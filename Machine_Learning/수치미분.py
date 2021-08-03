import numpy as np

### 수치미분 구현 ###

def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)     # 계산된 수치미분 값 저장 변수
                                # 입력 변수로 들어온 x 에 대한 미분값들을 같은 크기로 저장
    print("debug 1. initial input variable =", x)
    print("debug 2. initial grad =", grad)
    print("==========================================")
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        
        print("debug 3. idx =", idx, ", x[idx] = ", x[idx])
        
        tmp_val = x[idx]
        x[idx] = tmp_val + delta_x
        fx1 = f(x)
        
        x[idx] = tmp_val - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * delta_x)
        print("debug 4. grad[idx] =", grad[idx])
        print("debug 5. grad =", grad)
        print("==========================================")

        x[idx] = tmp_val
        it.iternext()
        
    return grad

def func1(inputt):
    x = inputt[0]
    y = inputt[1]
    return 2*x + 3*x*y + y**3

def func2(inputt):
    w = inputt[0, 0]
    x = inputt[0, 1]
    y = inputt[1, 0]
    z = inputt[1, 1]
    return (x*w + x*y*z + 3*w + z*y**2)

result_func1 = numerical_derivative(func1, np.array([3, 4], dtype = np.float64))

result_func2 = numerical_derivative(func2, np.array([[1, 2], [3, 4]], dtype = np.float64))

print('result_func1 :\n',result_func1)
print("")
print('result_func2 :\n',result_func2)
