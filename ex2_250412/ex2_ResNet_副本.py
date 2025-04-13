import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# 加载并预处理数据
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 标签转换为独热编码
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 数据增强
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
    ]
)



def residual_block(x, filters, strides=1, use_projection=False):
    input_tensor = x
    x = layers.Conv2D(filters, 3, strides=strides, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    if use_projection or strides != 1:
        input_tensor = layers.Conv2D(filters, 1, strides=strides, padding="same")(input_tensor)
        input_tensor = layers.BatchNormalization()(input_tensor)

    x = layers.Add()([x, input_tensor])
    x = layers.ReLU()(x)
    return x


# 构建 ResNet 模型
def resnet_cifar(input_shape=(32, 32, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual_block(x, 64, strides=1)
    x = residual_block(x, 64, strides=1)

    x = residual_block(x, 128, strides=2, use_projection=True)
    x = residual_block(x, 128, strides=1)

    x = residual_block(x, 256, strides=2, use_projection=True)
    x = residual_block(x, 256, strides=1)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


# 编译与训练配置
model = resnet_cifar()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 回调函数
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "cifar10_resnet.keras", save_best_only=True, monitor="val_accuracy"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

# 训练模型
history = model.fit(
    data_augmentation(x_train),
    y_train,
    batch_size=128,
    epochs=50,
    validation_split=0.1,
    callbacks=callbacks
)

# 评估与可视化
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# 保存模型
model_path = 'cifar10_ResNet_model.keras'
model_dir = os.path.dirname(model_path)
if model_dir and not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(model_path)
print(f"模型已保存到 {model_path}")