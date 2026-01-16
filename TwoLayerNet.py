import numpy as np
from collections import OrderedDict


from Layer import *
from deep_function import *


class TwoLayerNet:
    def __init__ ( self, input_size, hidden_size, output_size,
                weight_init_std=0.01 ) :
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn ( input_size, hidden_size )
        self.params['b1'] = np.zeros ( hidden_size )
        self.params['W2'] = weight_init_std * \
                            np.random.randn ( hidden_size, output_size )
        self.params['b2'] = np.zeros ( output_size )

        # 生成层
        #将神经网络保存到有序字典里面
        self.layers = OrderedDict ()
        self.layers['Affine1'] = \
            Affine ( self.params['W1'], self.params['b1'] )
        self.layers['Relu1'] = Relu ()
        self.layers['Affine2'] = \
            Affine ( self.params['W2'], self.params['b2'] )
        self.lastLayer = SoftmaxWithLoss ()


    def predict ( self, x ) :
        for layer in self.layers.values () :
            x = layer.forward ( x )
        return x

    # x:输入数据, t:监督数据
    def loss ( self, x, t ) :
        y = self.predict ( x )

        return cross_entropy_error ( y, t )

    def accuracy ( self, x, t ) :
        y = self.predict ( x )

        y = np.argmax ( y, axis=1 )
        t = np.argmax ( t, axis=1 )
        accuracy = np.sum ( y == t ) / float ( x.shape[0] )
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient ( self, x, t ) :
        self.loss ( x, t )
        # backward
        dout = 1
        dout = self.lastLayer.backward ( dout )
        layers = list ( self.layers.values () )
        layers.reverse ()
        for layer in layers :
            dout = layer.backward ( dout )
        #设计
        grads = {}
        grads['W1'] = numerical_gradient ( loss_W, self.params['W1'] )
        grads['b1'] = numerical_gradient ( loss_W, self.params['b1'] )
        grads['W2'] = numerical_gradient ( loss_W, self.params['W2'] )
        grads['b2'] = numerical_gradient ( loss_W, self.params['b2'] )
        return grads


