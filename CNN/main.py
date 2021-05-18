import tensorflow as tf
from tensorflow import keras

import os

import numpy as np
import matplotlib.pyplot as plt

EAGER = True

fashion_mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# plt.figure(figsize=(10, 10))  # 显示前25张图像
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#
# plt.show()

print(train_images.shape, train_labels.shape)

# 将图像的数据类型转换成浮点型，再将像素值缩小到0-1，完成数据的预处理
train_images = train_images.reshape([-1, 28, 28, 1]) / 255.0
test_images = test_images.reshape([-1, 28, 28, 1]) / 255.0

import numpy
train = numpy.loadtxt(open("./fashion-mnist_test_data.csv","rb"),delimiter=",",skiprows=0)
train=train.reshape([-1, 28, 28, 1]) / 255.0


model = keras.Sequential([

    keras.layers.Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=5, strides=1, padding='same'),

    keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
 
    keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),  # Padding method),

    keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

    keras.layers.Flatten(),
    
    keras.layers.Dense(256, activation=tf.nn.relu),
  
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

print(model.summary())

lr = 0.001
epochs = 21
# 编译模型
model.compile(optimizer=tf.optimizers.Adam(lr),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#训练模型
history=model.fit(train_images, train_labels, epochs=epochs,batch_size = 256, validation_data=(test_images[:1000], test_labels[:1000]))
print(history.history.keys())
import matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()



import pandas as pd

# 模型评测
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('the model\'s test_loss is {} and test_acc is {}'.format(test_loss, test_acc))

# 部分预测结果展示
show_images = test_images[:10]
print(show_images.shape)
predictions = model.predict(show_images)
predict_labels = np.argmax(predictions, 1)

plt.figure(figsize=(10, 5))  # 显示前10张图像，并在图像上显示类别
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.grid(False)
    plt.imshow(show_images[i, :, :, 0], cmap=plt.cm.binary)
    plt.title(class_names[predict_labels[i]])

plt.show()


print(test_images.shape)
predictions = model.predict(train)
predict_labels = np.argmax(predictions, 1)
print(predict_labels)
df = pd.DataFrame({'CLASS': predict_labels})

print(len(df))
df.to_csv("CNN.csv", index=False)