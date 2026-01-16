# coding: utf-8
import numpy as np


def identity_function ( x ) :
    return x


def step_function ( x ) :
    return np.array ( x > 0, dtype=np.int )


def sigmoid ( x ) :
    return 1 / (1 + np.exp ( -x ))


def sigmoid_grad ( x ) :
    return (1.0 - sigmoid ( x )) * sigmoid ( x )


def relu ( x ) :
    return np.maximum ( 0, x )


def relu_grad ( x ) :
    grad = np.zeros ( x )
    grad[x >= 0] = 1
    return grad


def softmax ( x ) :
    if x.ndim == 2 :
        x = x.T
        x = x - np.max ( x, axis=0 )
        y = np.exp ( x ) / np.sum ( np.exp ( x ), axis=0 )
        return y.T

    x = x - np.max ( x )  # 溢出对策
    return np.exp ( x ) / np.sum ( np.exp ( x ) )


def mean_squared_error ( y, t ) :
    return 0.5 * np.sum ( (y - t) ** 2 )


def cross_entropy_error ( y, t ) :
    if y.ndim == 1 :
        t = t.reshape ( 1, t.size )
        y = y.reshape ( 1, y.size )

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size :
        t = t.argmax ( axis=1 )
    batch_size = y.shape[0]
    return -np.sum ( np.log ( y[np.arange ( batch_size ), t] + 1e-7 ) ) / batch_size


def softmax_loss ( X, t ) :
    y = softmax ( X )
    return cross_entropy_error ( y, t )



def _numerical_gradient_1d ( f, x ) :
    h = 1e-4  # 0.0001
    grad = np.zeros_like ( x )

    for idx in range ( x.size ) :
        tmp_val = x[idx]
        x[idx] = float ( tmp_val ) + h
        fxh1 = f ( x )  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f ( x )  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient_2d ( f, X ) :
    if X.ndim == 1 :
        return _numerical_gradient_1d ( f, X )
    else :
        grad = np.zeros_like ( X )

        for idx, x in enumerate ( X ) :
            grad[idx] = _numerical_gradient_1d ( f, x )

        return grad


def numerical_gradient ( f, x ) :
    h = 1e-4  # 0.0001
    grad = np.zeros_like ( x )

    it = np.nditer ( x, flags=['multi_index'], op_flags=['readwrite'] )
    while not it.finished :
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float ( tmp_val ) + h
        fxh1 = f ( x )  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f ( x )  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext ()

    return grad