import numpy as np
import matplotlib.pyplot as plt
import sys, os
from dataset.mnist import load_mnist

sys.path.append(os.pardir)


#   均方误差
def mean_squared_error ( y, t ) :
    return 0.5 * np.sum(( y - t ) **2 )
#   交叉熵误差
def cross_entropy_error ( y, t ) :
    if y.ndim == 1 :
        t = np.reshape( t, t.size)
        y = np.reshape ( y, y.size )
    batch_size = y.shape[0]
    return -np.sum ( t * np.log ( y + 1e-7 ) ) / batch_size
#   数值微分
def numerical_diff( y , t ) :
    h = 1e-4
    return (f ( x + h ) - f ( x - h )) / (2 * h)
def function_1(x):
    return x[0]**2 + x[1]**2
# 梯度
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 生成和x形状相同的数组
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值
    return grad
#梯度法
#参数f是要进行最优化的函数，init_x是初始值，lr是学习率
#step_num是梯度法的重复次数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)
    print(x_train.shape) # (60000, 784)
    print(t_train.shape) # (60000, 10)

    train_size = x_train.shape[0]
    batch_size = 10
    #   随机选择想要的数字
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print(batch_mask)

    init_x = np.array ( [-3.0, 4.0] )
    x ,x_history= gradient_descent(function_1, init_x = init_x, lr=0.1 , step_num=100)
    print(x)
    plt.plot ( [-5, 5], [0, 0], '--b' )
    plt.plot ( [0, 0], [-5, 5], '--b' )
    plt.plot ( x_history[:, 0], x_history[:, 1], 'or' )

    plt.xlim ( -3.5, 3.5 )
    plt.ylim ( -4.5, 4.5 )
    plt.xlabel ( "X0" )
    plt.ylabel ( "X1" )
    plt.show ()
