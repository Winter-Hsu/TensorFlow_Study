import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

trainX = np.linspace(-1, 1, 100)
trainY = 2 * trainX + np.random.randn(*trainX.shape) * 0.3
# y=2x+噪声,randn产生n个随机数
# plt.plot(trainX, trainY, 'ro', label='original data')
# plt.legend()
# plt.show()

# ------------------create the model------------------------
# 占位符
X = tf.placeholder('float')
Y = tf.placeholder('float')

# define the model parameter
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

z = tf.multiply(X, W) + b

# --------------------反向优化-------------------------
cost = tf.reduce_mean(tf.square(Y - z))
learningRate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

# 定义参数
trainingEpochs = 20 # 迭代次数
displayStep = 2

# 启动session
with tf.Session() as sess:
    sess.run(init)
    plotdata = {'batchsize': [],'loss': []} #存放批次值和损失值
    # 向模型输入数据
    for epoch in range(trainingEpochs):
        for (x, y) in zip(trainX, trainY):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # 显示训练中的详细信息
        if epoch %displayStep == 0:
            loss = sess.run(cost, feed_dict={X: trainX, Y: trainY})
            print('Epoch:', epoch+1, 'cost=', loss, 'W=', sess.run(W), 'b=', sess.run(b))
            if not (loss == 'NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
    print('Finished!')
    print('cost=', sess.run(cost, feed_dict={X: trainX, Y: trainY}), 'W=', sess.run(W), 'b=', sess.run(b))
