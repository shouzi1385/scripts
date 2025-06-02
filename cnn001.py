from cProfile import label
from turtle import mode
import tensorflow as tf
from tensorflow import keras
import numpy as np

from keras import datasets,layers,models

import matplotlib.pyplot as plt
import os

(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()

train_images,test_images=train_images/255.0,test_images/255.0

class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# 模型保存路径
model_path = 'cifar10_cnn_model.h5'

# 检查模型是否已存在
if os.path.exists(model_path):
    # 加载已保存的模型
    print(f"加载已保存的模型: {model_path}")
    model = keras.models.load_model(model_path)
else:
    # 构建模型
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10))
    
    # 编译模型
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    
    # 训练模型
    print("开始训练模型...")
    history = model.fit(train_images,train_labels,epochs=10,validation_data=(test_images,test_labels))
    
    # 保存模型
    model.save(model_path)
    print(f"模型已保存至: {model_path}")

# 评估模型
test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)
print(f"测试准确率: {test_acc:.4f}")

# 预测新图像
img_path = 'cat1.png'  # 请确保图像存在，或替换为有效路径

# 检查图像是否存在
if not os.path.exists(img_path):
    print(f"错误：找不到图像文件 '{img_path}'")
else:
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(32, 32))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # 归一化

        # 预测
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        # 显示结果
        plt.figure(figsize=(6, 4))
        plt.imshow(img)
        plt.title(f'预测结果: {class_names[predicted_class]}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"处理图像时出错: {e}")