在本篇文章中，用手刻Adagrad完成了「机器学习-李宏毅」的HW1-预测PM2.5的作业。其中包括对数据的处理，训练模型，预测，并使用sklearn toolkit的结果进行比较。
有关HW1的相关数据、源代码、预测结果等，欢迎光临小透明的[GitHub](https://github.com/f1ed/ML-HW1)  

<!--more-->
# Task Description

[kaggle link](https://www.kaggle.com/c/ml2020spring-hw1) 

从中央气象局网站下载的真实观测资料，必须利用linear regression或其他方法预测PM2.5的值。

观测记录被分为train set 和 test set, 前者是每个月前20天所有资料；后者是从剩下的资料中随机取样出来的。

train.csv: 每个月前20天的完整资料。

test.csv: 从剩下的10天资料中取出240笔资料，每一笔资料都有连续9小时的观测数据，必须以此观测出第十小时的PM2.5.

# Process Data

train data如下图，每18行是一天24小时的数据，每个月取了前20天（时间上是连续的小时）。

[![GyqsyR.md.png](https://s1.ax1x.com/2020/04/06/GyqsyR.md.png)](https://imgchr.com/i/GyqsyR) 

test data 如下图，每18行是一笔连续9小时的数据，共240笔数据。

[![Gyqcex.md.png](https://s1.ax1x.com/2020/04/06/Gyqcex.md.png)](https://imgchr.com/i/Gyqcex) 

----

1. **最大化training data size**

   每连续10小时的数据都是train set的data。为了得到更多的data，应该把每一天连起来。即下图这种效果：

   [![GyqyO1.md.png](https://s1.ax1x.com/2020/04/06/GyqyO1.md.png)](https://imgchr.com/i/GyqyO1) 

   每个月就有： $20*24-9=471$ 笔data

   ``` py
   # Dictionary: key:month value:month data
   month_data = {}
   
   # make data timeline continuous
   for month in range(12):
       temp = np.empty(shape=(18, 20*24))
       for day in range(20):
           temp[:, day*24: (day+1)*24] = data[(month*20+day)*18: (month*20+day+1)*18, :]
       month_data[month] = temp
   ```

2. **筛选需要的Features** :

   这里，我就只考虑前9小时的PM2.5，当然还可以考虑和PM2.5等相关的氮氧化物等feature。

   **training data** 

   ``` pyt
   # x_data v1: only consider PM2.5
   x_data = np.empty(shape=(12*471, 9))
   y_data = np.empty(shape=(12*471, 1))
   for month in range(12):
       for i in range(471):
           x_data[month*471+i][:] = month_data[month][9][i: i+9]
           y_data[month*471+i] = month_data[month][9][i+9]
   ```

   **testing data** 

   ``` py
   # Testing data features
   test_x = np.empty(shape=(240, 9))
   for day in range(240):
       test_x[day, :] = test_data[18*day+9, :]
   test_x = np.concatenate((np.ones(shape=(240, 1)), test_x), axis=1)
   ```

3. **Normalization** 

   ``` py
   # feature scale: normalization
   mean = np.mean(x_data, axis=0)
   std = np.std(x_data, axis=0)
   for i in range(x_data.shape[0]):
       for j in range(x_data.shape[1]):
           if std[j] != 0:
               x_data[i][j] = (x_data[i][j] - mean[j]) / std[j]
   
   for i in range(test_data.shape[0]):
       for j in range(test_data.shape[1]):
           if std[j] != 0:
               test_data[i][j] = (test_data[i][j] - mean[j])/std[j]
   ```



# Training 

手刻Adagrad 进行training。（挖坑：RMSprop、Adam[1]

**Linear Pseudo code** 

``` pseudocode
 Declare weight vector, initial lr ,and # of iteration
 for i_th iteration :
 	 y’ = the product of train_x  and weight vector
 	 Loss = y’ - train_y
 	 gradient = 2*np.dot((train_x)’, Loss )
    weight vector -= learning rate * gradient
```

其中的矩阵操作时，注意求gradient时矩阵的维度。可参考下图。

[![Gyqgw6.md.png](https://s1.ax1x.com/2020/04/06/Gyqgw6.md.png)](https://imgchr.com/i/Gyqgw6) 



**Adagrad Pseudo code**

``` ps
Declare weight vector, initial lr ,and # of iteration
Declare prev_gra storing gradients in every previous iterations
 for i_th iteration :
 	 y’ = the inner product of train_x  and weight vector
 	 Loss = y’ - train_y
 	 gradient = 2*np.dot((train_x)’, Loss )
        prev_gra += gra**2
	 	ada = np.sqrt(prev_gra)
    weight vector -= learning rate * gradient / ada
```

注：代码实现时，将bias存在w[0]处，x_data的第0列全1。因为w和b可以一同更新。（当然，也可以分开更新）

**Adagrad training** 

``` py
# train-adagrad

batch = x_data.shape[0]  # full batch
epoch = 400

# some parameters for training
dim = x_data.shape[1]+1
w = np.zeros(shape=(dim, 1))  # concatenate bias = w[0]
lr = np.full((dim, 1), 0.8)  # learning rate
grad = np.empty(shape=(dim, 1))  # gradient of loss to every para
gradsum = np.zeros(shape=(dim, 1))  # sum of gradient**2

x_data = np.concatenate((np.ones(shape=(x_data.shape[0], 1)), x_data), axis=1)

loss_his = np.empty(shape=(epoch, 1))
for T in range(epoch):
    L = y_data - np.dot(x_data, w)
    loss_his[T] = np.sum(L**2) / x_data.shape[0]
    grad = (-2)*np.dot(np.transpose(x_data), L)
    gradsum = gradsum + grad**2
    w = w - lr*grad/(gradsum**0.5)
```



# Testing

```py
answer = np.dot(test_x, w)
```



# Draw and Analysis

在每次迭代更新时，我将Loss的值存了下来，以便可视化Loss的变化和更新速度。

Loss的变化如下图：(红色的是sklearn toolkit的loss结果)

[![Gyqrl9.png](https://s1.ax1x.com/2020/04/06/Gyqrl9.png)](https://imgchr.com/i/Gyqrl9) 

此外，在源代码中，使用sklearn toolkit来比较结果。

结果如下：

``` py
v1: only consider PM2.5

Using sklearn
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
bias= [21.37402689]
w= [[ 0.00000000e+00]
 [-5.54801503e-01]
 [-4.32873874e-01]
 [ 3.63669814e+00]
 [-3.99037687e+00]
 [-9.07364636e-01]
 [ 8.83495803e+00]
 [-9.51785135e+00]
 [ 1.32734655e-02]
 [ 1.81886444e+01]]

In our model
bias= [19.59387132]
w= [[-0.14448468]
 [ 0.39205748]
 [ 0.26897134]
 [-1.02415371]
 [ 1.21151411]
 [ 2.21925424]
 [-5.48242478]
 [ 4.01080346]
 [13.56369122]]
```

发现参数有一定差异，于是我在testing时，也把sklearn的结果进行预测比较。

一部分结果如下：

```py
['id', 'value', 'sk_value']
['id_0', 3.551092352912313, 5.37766865368331]
['id_1', 13.916795471648756, 16.559245678900034]
['id_2', 24.811333478647043, 23.5085950470451]
['id_3', 5.101440436158914, 6.478306159981166]
['id_4', 26.7374726797937, 27.207516152986663]
['id_5', 19.43735346531517, 21.916809502961648]
['id_6', 22.20460696285646, 24.751295357256392]
['id_7', 29.660872382552682, 30.24344042612033]
['id_8', 17.5964527734513, 16.64242443764712]
['id_9', 56.58017426943178, 59.760988216575115]
['id_10', 13.767504260132299, 10.808372404511037]
['id_11', 11.743000466164233, 11.526958393801682]
['id_12', 59.509878887026105, 64.201008247897]
['id_13', 53.19824337746267, 54.3856368053018]
['id_14', 21.97191108867921, 24.530720709840974]
['id_15', 10.833283625735444, 14.350345549104446]
```

# Code

有关HW1的相关数据、源代码、预测结果等，欢迎光临小透明的[GitHub](https://github.com/f1ed/ML-HW1) 

``` py
#########################
# Date: 2020-4-4
# Author: FredLau
# HW1: predict the PM2.5
##########################

import sys
import numpy as np
import pandas as pd
import csv
from sklearn import linear_model
import matplotlib.pyplot as plt

#####################
# process data

# process train data
raw_data = np.genfromtxt('data/train.csv', delimiter=',')
data = raw_data[1:, 3:]
data[np.isnan(data)] = 0  # process nan

# Dictionary: key:month value:month data
month_data = {}

# make data timeline continuous
for month in range(12):
    temp = np.empty(shape=(18, 20*24))
    for day in range(20):
        temp[:, day*24: (day+1)*24] = data[(month*20+day)*18: (month*20+day+1)*18, :]
    month_data[month] = temp

# x_data v1: only consider PM2.5
x_data = np.empty(shape=(12*471, 9))
y_data = np.empty(shape=(12*471, 1))
for month in range(12):
    for i in range(471):
        x_data[month*471+i][:] = month_data[month][9][i: i+9]
        y_data[month*471+i] = month_data[month][9][i+9]

# process test data
test_raw_data = np.genfromtxt('data/test.csv', delimiter=',')
test_data = test_raw_data[:, 2:]
test_data[np.isnan(test_data)] = 0

# feature scale: normalization
mean = np.mean(x_data, axis=0)
std = np.std(x_data, axis=0)
for i in range(x_data.shape[0]):
    for j in range(x_data.shape[1]):
        if std[j] != 0:
            x_data[i][j] = (x_data[i][j] - mean[j]) / std[j]

for i in range(test_data.shape[0]):
    for j in range(test_data.shape[1]):
        if std[j] != 0:
            test_data[i][j] = (test_data[i][j] - mean[j])/std[j]

# Testing data features
test_x = np.empty(shape=(240, 9))
for day in range(240):
    test_x[day, :] = test_data[18*day+9, :]
test_x = np.concatenate((np.ones(shape=(240, 1)), test_x), axis=1)

################################
# train-adagrad

batch = x_data.shape[0]  # full batch
epoch = 400

# some parameters for training
dim = x_data.shape[1]+1
w = np.zeros(shape=(dim, 1))  # concatenate bias = w[0]
lr = np.full((dim, 1), 0.8)  # learning rate
grad = np.empty(shape=(dim, 1))  # gradient of loss to every para
gradsum = np.zeros(shape=(dim, 1))  # sum of gradient**2

x_data = np.concatenate((np.ones(shape=(x_data.shape[0], 1)), x_data), axis=1)

loss_his = np.empty(shape=(epoch, 1))
for T in range(epoch):
    L = y_data - np.dot(x_data, w)
    loss_his[T] = np.sum(L**2) / x_data.shape[0]
    grad = (-2)*np.dot(np.transpose(x_data), L)
    gradsum = gradsum + grad**2
    w = w - lr*grad/(gradsum**0.5)

f = open('output/v1.csv', 'w')
sys.stdout = f
print('v1: only consider PM2.5\n')

###############################
# train by sklearn linear model
print('Using sklearn')
reg = linear_model.LinearRegression()
print(reg.fit(x_data, y_data))
print('bias=', reg.intercept_)
print('w=', reg.coef_.transpose())
print('\n')

# In our model
print('In our model')
print('bias=', w[0])
print('w=', w[1:])

###########################
# draw change of loss
plt.xlim(0, epoch)
plt.ylim(0, 10)
plt.xlabel('$iteration$', fontsize=16)
plt.ylabel('$Loss$', fontsize=16)
iteration = np.arange(0, epoch)
plt.plot(iteration, loss_his/100, '-', ms=3, lw=2, color='black')


sk_w = reg.coef_.transpose()
sk_w[0] = reg.intercept_
sk_loss = np.sum((y_data - np.dot(x_data, sk_w))**2) / x_data.shape[0]
plt.hlines(sk_loss/100, 0, epoch, colors='red', linestyles='solid')
plt.legend(['adagrad', 'sklearn'])
plt.show()
# plt.savefig('output/v1.png')
f.close()

##############
# test (sklearn vs our adagrad
f = open('output/v1test.csv', 'w')
sys.stdout = f

title = ['id', 'value', 'sk_value']
answer = np.dot(test_x, w)
sk_answer = np.dot(test_x, sk_w)
print(title)
for i in range(test_x.shape[0]):
    content = ['id_'+str(i), answer[i][0], sk_answer[i][0]]
    print(content)

f.close()
```



# Reference

1. 待完成