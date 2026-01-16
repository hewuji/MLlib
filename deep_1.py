from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle


# 定义激活函数
def sigmoid ( x ) :
    return 1 / (1 + np.exp ( -x ))


def softmax ( a ) :
    c = np.max ( a )
    exp_a = np.exp ( a - c )  # 溢出对策
    sum_exp_a = np.sum ( exp_a )
    y = exp_a / sum_exp_a
    return y


# 直接使用Keras内置的MNIST数据集加载函数
def load_mnist ( flatten=True, normalize=False ) :
    """
    加载MNIST数据集
    - flatten: 是否将图像展平为一维数组 (28x28 -> 784)
    - normalize: 是否将像素值归一化到0-1之间
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data ()

    if flatten :
        x_train = x_train.reshape ( -1, 784 )
        x_test = x_test.reshape ( -1, 784 )

    if normalize :
        x_train = x_train.astype ( 'float32' ) / 255.0
        x_test = x_test.astype ( 'float32' ) / 255.0
    else :
        x_train = x_train.astype ( 'float32' )
        x_test = x_test.astype ( 'float32' )

    return (x_train, y_train), (x_test, y_test)


def img_show ( img ) :
    """显示图像"""
    # 如果图像是一维的，先转换为二维
    if img.ndim == 1 :
        img = img.reshape ( 28, 28 )

    # 如果是归一化的数据，转换回0-255范围
    if img.max () <= 1.0 :
        img = (img * 255).astype ( np.uint8 )

    pil_img = Image.fromarray ( np.uint8 ( img ) )
    pil_img.show ()


def get_data () :
    (x_train, y_train), (x_test, t_test) = \
        load_mnist ( normalize=True, flatten=True )
    # 修正：返回 t_test 而不是 y_test
    return x_test, t_test


def init_network () :
    with open ( "sample_weight.pkl", 'rb' ) as f :
        network = pickle.load ( f )
    return network


def predict ( network, x ) :
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot ( x, W1 ) + b1
    z1 = sigmoid ( a1 )
    a2 = np.dot ( z1, W2 ) + b2
    z2 = sigmoid ( a2 )
    a3 = np.dot ( z2, W3 ) + b3
    y = softmax ( a3 )
    return y


# 主程序
if __name__ == "__main__" :
    # 加载数据
    (x_train, y_train), (x_test, t_test) = load_mnist ( flatten=True, normalize=False )

    print ( f"训练集数据形状: {x_train.shape}" )  # (60000, 784)
    print ( f"训练集标签形状: {y_train.shape}" )  # (60000,)
    print ( f"测试集数据形状: {x_test.shape}" )  # (10000, 784)
    print ( f"测试集标签形状: {t_test.shape}" )  # (10000,)

    # 显示第一张图片
    img = x_train[0]
    label = y_train[0]
    print ( f"标签: {label}" )  # 5
    print ( f"图像形状: {img.shape}" )  # (784,)

    # 重塑为28x28显示
    img_reshaped = img.reshape ( 28, 28 )
    print ( f"重塑后形状: {img_reshaped.shape}" )  # (28, 28)


    img_show(img)

    # 获取测试数据
    x, t = get_data ()  # x是测试集数据，t是测试集标签

    try :
        # 初始化网络
        network = init_network ()

        # 批量处理提高效率
        batch_size = 100  # 批数量
        accuracy_cnt = 0

        # 注意：这里我们使用测试集进行预测
        for i in range ( 0, len ( x ), batch_size ) :
            x_batch = x[i :i + batch_size]
            y_batch = predict ( network, x_batch )
            p = np.argmax ( y_batch, axis=1 )
            accuracy_cnt += np.sum ( p == t[i :i + batch_size] )

        print ( f"Accuracy: {float ( accuracy_cnt ) / len ( x ):.4f}" )

    except FileNotFoundError :
        print ( "错误：未找到 'sample_weight.pkl' 文件" )
        print ( "请确保 sample_weight.pkl 文件在当前目录下" )
    except Exception as e :
        print ( f"发生错误: {e}" )