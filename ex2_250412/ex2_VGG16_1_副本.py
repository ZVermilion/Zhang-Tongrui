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

# 搭建 VGG16 模型
def VGG16(num_classes=10):
    model = keras.Sequential([
        # 第一段卷积层
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # 第二段卷积层
        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # 第三段卷积层
        keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # 第四段卷积层
        keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # 第五段卷积层
        keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # 全连接层
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 创建 VGG16 模型实例
model = VGG16()

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_data, train_label, epochs=10, validation_data=(test_data, test_label))

# 绘制损失和精度图
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train loss', 'Validation loss'])
plt.show()

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.show()

# 保存模型
model_path = 'cifar10_VGG16_model.keras'
model_dir = os.path.dirname(model_path)
if model_dir and not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(model_path)
print(f"模型已保存到 {model_path}")