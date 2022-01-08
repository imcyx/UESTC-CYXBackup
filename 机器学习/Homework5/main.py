import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 导入模块，生成模拟数据集
import tensorflow as tf

filenames = ["abalone_test.data", "abalone_train.data"]

def preprocess(file):
    def filter_n(s):
        ss = tf.constant([-1, ], dtype=tf.int32)
        if tf.equal(s, "M"):
            ss = tf.constant([1, 0, 0])
        elif tf.equal(s, "F"):
            ss = tf.constant([0, 1, 0])
        elif tf.equal(s, "I"):
            ss = tf.constant([0, 0, 1])
        return ss

    # 加载过滤后的数据集
    paths = list(tf.data.Dataset.list_files(file))
    sentences_ds = tf.data.Dataset.from_tensor_slices(paths)
    sentences_ds = sentences_ds.interleave(
        lambda text_file: tf.data.TextLineDataset(text_file).filter(
            lambda line: tf.not_equal(tf.strings.substr(line, 0, 1), "%")).filter(
            lambda line: tf.not_equal(tf.strings.substr(line, 0, 1), "")))

    X = sentences_ds.map(lambda text: tf.strings.to_number(tf.strings.split(text, sep=",")[1:]))
    Y = sentences_ds.map(lambda text: tf.strings.split(text, sep=",")[:1]).map(filter_n)
    return X, Y

X, Y = preprocess('abalone_train.data')
train_all =  tf.data.Dataset.zip((X, Y)).shuffle(len(list(X)))
datasets = train_all.batch(128)
sample = list(datasets.as_numpy_iterator())
train_all = train_all.batch(len(list(X)))
train_all = list(train_all.as_numpy_iterator())[0]
print(train_all[0])

X, Y = preprocess('abalone_test.data')
test_all = tf.data.Dataset.zip((X, Y)).shuffle(len(list(X)))
test_all = test_all.batch(len(list(X)))
test_all = list(test_all.as_numpy_iterator())[0]

tf.compat.v1.disable_eager_execution()
# 定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 8))
y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))

w1 = tf.Variable(tf.random.normal([8, 5], stddev=1, seed=1, dtype=tf.float32))
w2 = tf.Variable(tf.random.normal([5, 3], stddev=1, seed=1, dtype=tf.float32))
w3 = tf.Variable(tf.random.normal([3, 3], stddev=1, seed=1, dtype=tf.float32))
b1 = tf.Variable(1.0, dtype = tf.float32) #定义偏差值
b2 = tf.Variable(1.0, dtype = tf.float32) #定义偏差值
b3 = tf.Variable(1.0, dtype = tf.float32) #定义偏差值

a1 = tf.add(tf.matmul(x, w1), b1)
a2 = tf.add(tf.matmul(a1, w2), b2)
y = tf.nn.leaky_relu(tf.add(tf.matmul(a2, w3), b3))

# 定义损失函数及反向传播方法
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.compat.v1.train.AdamOptimizer(0.005).minimize(loss)
# 计算预测值，准确数
pred = tf.argmax(y, axis=1)
pred = tf.cast(pred, dtype=tf.int64)
correct = tf.equal(pred, tf.argmax(y_, axis=1))
correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

# 生成会话，训练STEPS轮
with tf.compat.v1.Session() as sess:
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    # 输出目前（未经训练）的参数取值
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))

    # 训练模型
    STEPS = 1000
    for i in range(STEPS):
        for num in sample:
            sess.run(train_step, feed_dict={x: num[0], y_: num[1]})
        # if i % 500 == 0:
        total_loss = sess.run(loss, feed_dict={x: train_all[0], y_: train_all[1]})
        print("After %d training step(s), loss on all data is %g" % (i, total_loss))

    # 输出训练后的参数取值
    # print("w1:", sess.run(w1))
    # print("w2:", sess.run(w2))
    # 输出判断正确数目和正确率
    print(sess.run([correct, correct/len(test_all[0])], feed_dict={x: test_all[0], y_: test_all[1]}))

