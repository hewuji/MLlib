import pandas as pd
from sklearn.model_selection import train_test_split #导⼊sklearn⼯具库
from sklearn.linear_model import LinearRegression  #导⼊线性回归算法模型
import matplotlib.pyplot as plt #导⼊Matplotlib库


df_housing = pd.read_csv ( "C:/Users/Administrator/Desktop/machine/MLlib/rate.csv" )
print ( df_housing.head () )
#构建特征集X
X = df_housing.drop("median_house_value", axis=1)
#构建特征集y
y = df_housing["median_house_value"]
#以80%/20%的⽐例进⾏数据集的拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#确定线性回归算法
model = LinearRegression()
#根据训练集数据, 训练机器, 拟合函数
model.fit(X_train, y_train)
#预测验证集的y值
y_pred = model.predict(X_test)
print ('房价的真值(测试集)', y_test)
print ('预测的房价(测试集)', y_pred)
# 评估预测结果
print("给预测评分：", model.score(X_test, y_test))
#⽤散点图显示家庭收⼊中位数和房价中位数的分布
plt.plot( X_test.median_income, y_test, color='brown')
#画出回归函数(从特征到预测标签)
plt.plot(X_test.median_income, y_pred, color='green', linewidth=1)
#x轴：家庭收⼊中位数
plt.xlabel('Median Income')
#y轴：房价中位数
plt.ylabel('Median House Value')
#显示房价分布和机器学习到的函数模型
plt.show()