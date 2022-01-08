import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_x = np.array(
    [[4, 171.2],
     [4, 174.2],
     [5, 204.3],
     [4, 218.7],
     [4, 219.4],
     [7, 240.4],
     [4, 273.5],
     [5, 294.8],
     [10, 330.2],
     [7, 333.1],
     [5, 366.0],
     [6, 350.9],
     [4, 357.9],
     [5, 359.0],
     [7, 371.9],
     [9, 435.3],
     [8, 523.9],
     [10, 604.1],], dtype=np.float32
)
train_y = np.array(
    [[450.5],
     [507.7],
     [613.9],
     [563.4],
     [501.5],
     [781.5],
     [541.8],
     [611.1],
     [1222.1],
     [793.2],
     [660.8],
     [792.7],
     [580.8],
     [612.7],
     [890.8],
     [1121.0],
     [1094.2],
     [1253.0],], dtype=np.float32
)


tf.compat.v1.disable_eager_execution()
# 定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random.normal([2, 1], name="weight1"))
w2 = tf.Variable(tf.random.normal([2, 1], name="weight2"))
b1 = tf.Variable(1.0, name="bias") #定义偏差值

# y = tf.matmul(x, w1) + tf.matmul(tf.pow(x, 2), w2) + b1
y = tf.matmul(x, w1) + b1

# 定义损失函数及反向传播方法
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.compat.v1.train.AdamOptimizer(0.1).minimize(loss)

s = tf.range(13)
# 生成会话，训练STEPS轮
with tf.compat.v1.Session() as sess:
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    # 输出目前（未经训练）的参数取值
    print("w:", sess.run(w1))
    print("b:", sess.run(b1))

    # 训练模型
    STEPS = 20000
    for i in range(STEPS):
        sess.run(train_step, feed_dict={x: train_x, y_: train_y})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: train_x, y_: train_y})
            print("After %d training step(s), loss on all data is %g" % (i, total_loss))

    # 输出最终训练权重和偏置
    print(sess.run(loss, feed_dict={x: train_x, y_: train_y}))
    weight1, weight2, bias = sess.run([w1, w2, b1])
    print(f'weight1:{weight1}\nweight2:{weight2}\nbias:{bias}')

    a = sess.run(y, feed_dict={x: train_x, w1: weight1, w2: weight2, b1: bias})
    plt.plot(np.arange(0, len(train_x)).astype(dtype=np.str), a, label="Fitted_line")
    plt.plot(np.arange(0, len(train_x)).astype(dtype=np.str), train_y, 'ro', label="Output")

    plt.text(0, 1000, f'y=x*w+b\nW: {weight1}\nb: {float(int(bias*1000)/1000)}', fontsize=12, color='purple')
    plt.xlabel("Homework 3-2")
    plt.legend()
    plt.show()
