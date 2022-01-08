import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_x = np.array(
    [[7, 26, 6, 60],
     [1, 29, 15, 52],
     [11, 56, 8, 20],
     [11, 31, 8, 47],
     [7, 52, 6, 33],
     [11, 55, 9, 22],
     [3, 71, 17, 6],
     [1, 31, 22, 44],
     [2, 54, 18, 22],
     [21, 47, 4, 26],
     [1, 40, 23, 34],
     [11, 66, 9, 12],
     [10, 68, 8, 12]], dtype=np.float32
)
train_y = np.array(
    [[78.5],
     [74.3],
     [104.3],
     [87.6],
     [95.9],
     [109.2],
     [102.7],
     [72.5],
     [93.1],
     [115.9],
     [83.8],
     [113.3],
     [109.4]], dtype=np.float32
)


tf.compat.v1.disable_eager_execution()
# 定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random.normal([4, 1], name="weight"))
b1 = tf.Variable(1.0, name="bias") #定义偏差值

y = tf.add(tf.matmul(x, w1), b1)

# 定义损失函数及反向传播方法
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.compat.v1.train.AdamOptimizer(0.5).minimize(loss)

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
    weight, bias = sess.run([w1, b1])
    print(weight, bias)

    print(tf.add(tf.matmul(train_x, weight),bias).eval(), train_y)
    plt.plot(range(0,13), tf.add(tf.matmul(train_x, weight),bias).eval(), label="Fitted_line")
    plt.plot(range(0,13), train_y, 'ro', label="Output")
    plt.text(2.5, 72, f'y=x*w+b\nW: {weight}\nb: {float(int(bias*1000)/1000)}', fontsize=12, color='purple')
    plt.xlabel("Homework 3-1")
    plt.legend()
    plt.show()
