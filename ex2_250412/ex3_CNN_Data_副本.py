import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import os

# 加载数据集
(train_data, train_label), (test_data, test_label) = cifar10.load_data()

# 数据归一化
train_data = train_data.astype('float32') / 255.
test_data = test_data.astype('float32') / 255.


# 标签预处理
def one_hot(label, num_classes):
    label_one_hot = np.eye(num_classes)[label]
    return label_one_hot


num_classes = 10
train_label = train_label.astype('int32')
train_label = np.squeeze(train_label)
train_label = one_hot(train_label, num_classes)
test_label = test_label.astype('int32')
test_label = np.squeeze(test_label)
test_label = one_hot(test_label, num_classes)

# 数据增强
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.2)
])

# 创建 tf.data.Dataset 对象
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))

# 应用数据增强并进行缓存和预取
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32)
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(32).cache().prefetch(tf.data.AUTOTUNE)

# 构建网络 cnn
from keras import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout

cnn = Sequential()
# unit1
cnn.add(Convolution2D(32, kernel_size=[3, 3], input_shape=(32, 32, 3), activation='relu',
                      padding='same'))
cnn.add(Convolution2D(32, kernel_size=[3, 3], activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=[2, 2], padding='same'))
cnn.add(Dropout(0.5))
# unit2
cnn.add(Convolution2D(64, kernel_size=[3, 3], activation='relu', padding='same'))
cnn.add(Convolution2D(64, kernel_size=[3, 3], activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=[2, 2], padding='same'))
cnn.add(Dropout(0.5))
cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))
cnn.summary()

# 编译模型
cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy', metrics=['acc'])

# 训练模型
history_cnn = cnn.fit(train_dataset, epochs=50, validation_data=test_dataset, verbose=1)

# 绘制损失和精度图
plt.figure(1)
plt.plot(np.array(history_cnn.history['loss']))
plt.plot(np.array(history_cnn.history['val_loss']))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['loss', 'val_loss'])
plt.show()

plt.figure(2)
plt.plot(np.array(history_cnn.history['acc']))
plt.plot(np.array(history_cnn.history['val_acc']))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['acc', 'val_acc'])
plt.show()

# 保存模型
model_path = 'cifar10_cnn_model_Data.keras'
model_dir = os.path.dirname(model_path)
if model_dir and not os.path.exists(model_dir):
    os.makedirs(model_dir)
cnn.save(model_path)
print(f"模型已保存到 {model_path}")
