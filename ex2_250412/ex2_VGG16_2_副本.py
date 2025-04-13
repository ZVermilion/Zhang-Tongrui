import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
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
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(train_data)

# 加载预训练的 VGG16 模型
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
for layer in base_model.layers:
    layer.trainable = False

# 添加自定义层
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 创建新模型
model = Model(inputs=base_model.input, outputs=predictions)


# 学习率衰减策略
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 10:
        lr = 0.0001
    if epoch > 20:
        lr = 0.00001
    return lr


# 编译模型
model.compile(optimizer=Adam(learning_rate=lr_schedule(0)),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
lr_scheduler = LearningRateScheduler(lr_schedule)
history = model.fit(datagen.flow(train_data, train_label, batch_size=32),
                    steps_per_epoch=len(train_data) // 32,
                    epochs=50,
                    validation_data=(test_data, test_label),
                    callbacks=[lr_scheduler])

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
model_path = 'cifar10_VGG16_model_improved2.keras'
model_dir = os.path.dirname(model_path)
if model_dir and not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(model_path)
print(f"模型已保存到 {model_path}")
