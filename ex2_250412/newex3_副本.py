import tensorflow as tf
from tensorflow import keras
import numpy as np

# 加载保存的模型
model_path = 'cifar10_cnn_model_50.keras'
cnn = keras.models.load_model(model_path)
print("模型已加载")
from keras.datasets import cifar10
(_, _), (test_data, test_label) = cifar10.load_data()
x_test = test_data.astype('float32') / 255.
def one_hot(label, num_classes):
    label_one_hot = np.eye(num_classes)[label]
    return label_one_hot

num_classes = 10
test_label = test_label.astype('int32')
test_label = np.squeeze(test_label)
y_test = one_hot(test_label, num_classes)
test_out = cnn.predict(x_test)
correct_predictions = 0
total_predictions = x_test.shape[0]

for i in range(total_predictions):
    predicted_class = np.argmax(test_out[i])
    true_class = np.argmax(y_test[i])
    if predicted_class == true_class:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print(f"预测准确率: {accuracy * 100:.2f}%")