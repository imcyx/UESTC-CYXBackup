import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import  tensorflow as tf
from    tensorflow.keras import Input, layers, optimizers, Sequential, metrics

def preprocess(data_name):
    x, y = [], []
    data_y = None
    with open(data_name, "r") as test_datasets:
        for data in test_datasets.readlines():
            data = data.strip("\n")
            if data:
                data = data.split(",")
                if data[-1] == 'Iris-setosa':
                    data_y = 0
                elif data[-1] == 'Iris-versicolor':
                    data_y = 1
                elif data[-1] == 'Iris-virginica':
                    data_y = 2
                data = [float(data_) for data_ in data[:-1]]
            else:
                continue
            x.append(data)
            y.append(data_y)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    return x, y


def train():
    x, y = preprocess("iris_train.data")
    x_val, y_val = preprocess("iris_test.data")
    y = tf.one_hot(y, depth=3) # [50k, 10]
    y_val = tf.one_hot(y_val, depth=3) # [10k, 10]
    print('datasets:', x.shape, y.shape, x_val.shape, y_val.shape)

    train_db = tf.data.Dataset.from_tensor_slices((x,y))
    train_db = train_db.shuffle(x.shape[0]).batch(15)
    test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_db = test_db.batch(15)

    sample = next(iter(train_db))
    print('batch:', sample[0].shape, sample[1].shape)

    network = Sequential([layers.Dense(32, activation='relu'),
                          layers.Dense(16, activation='relu'),
                          layers.Dense(8, activation='relu'),
                          layers.Dense(3)])
    network.build(input_shape=(15, 4))
    network.summary()

    network.compile(optimizer=optimizers.Adam(learning_rate=0.01),
                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                    )

    network.fit(train_db, epochs=10, validation_data=test_db, validation_freq=2)

    # network.evaluate(test_db)
    sample = next(iter(test_db))
    x = sample[0]
    y = sample[1]  # one-hot
    pred = network.predict(x)  # [b, 3]
    # convert back to number
    y = tf.argmax(y, axis=1)
    pred = tf.argmax(pred, axis=1)
    print(pred)
    print(y)

if __name__ == '__main__':
    train()