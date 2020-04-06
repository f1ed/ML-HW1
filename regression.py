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