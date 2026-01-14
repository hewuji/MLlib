import matplotlib.pyplot as plt
import numpy as np


#   读入训练数据
train = np.loadtxt('rate.csv', delimiter=',', skiprows=1)
train_x = train[:,0]
train_y = train[:,1]
#   参数初始化
theta0 = np.random.rand()
theta1 = np.random.rand()
#   预测函数
def f(x):
    return theta0 + theta1 * x
#   目标函数
def E(v,y):
    return 0.5 * np.sum((y - f(v)) ** 2)
# 标准化
#平均值
mu = train_x.mean()
#标准差
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma
train_z = standardize(train_x)
# 学习率
ETA = 1e-7
# 添加最大迭代次数防止无限循环
max_iter = 500
# 误差的差值
diff = 1
# 更新次数
count = 0
# 重复学习
error = E(train_z, train_y)
while diff > 1e-2:
    # 更新结果保存到临时变量
    tmp0 = theta0 - ETA * np.sum ( (f ( train_z ) - train_y) )
    tmp1 = theta1 - ETA * np.sum ( (f ( train_z ) - train_y) * train_z )
    # 更新参数
    theta0 = tmp0
    theta1 = tmp1
    # 计算与上⼀次误差的差值
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error
    # 输出⽇志
    count += 1
    log = ' 第{} 次: theta0 = {:.3f}, theta1 = {:.3f}, 差值 = {:.4f}'
    print(log.format(count, theta0, theta1, diff))
print(f'训练完成，共迭代{count}次')
x = np.linspace(-3, 3, 100)
print(f(standardize(10000)))
#绘图
plt.plot(train_z, train_y,'o')
plt.plot(x, f(x))
plt.show()