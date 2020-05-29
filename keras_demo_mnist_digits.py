# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:04:03 2020
DNN and CNN on keras.datasets.mnist
@author: lwang
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print ('tf:', tf.__version__)

#%% Get the data as Numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test_orig = x_test 
y_test_integer = y_test # keep for evaluating prediction

#%% show some samples
# pick a sample to plot
sample = 1
image = x_train[sample]
# plot the sample
fig = plt.figure
plt.imshow(image, cmap='gray')
# plt.show()

# pick first N samples
num = 10
images = x_train[:num]
labels = y_train[:num]
num_row = 2
num_col = 5
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()

#%% reshape data: (x_train, y_train), (x_test, y_test)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
# convert to one-hot class label
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# tensorflow dataset
batch_size = 128 # how many samples to fetch each time from dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


#%% Build a model
start_time = time.time()
tf.keras.backend.clear_session()

# (A) baseline model: MLP
# inputs = keras.Input(shape=(28, 28))
# x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
# x = layers.Flatten()(x)
# x = layers.Dense(128, activation="relu")(x)
# x = layers.Dense(128, activation="relu")(x)
# outputs = layers.Dense(10, activation="softmax")(x)
# model = keras.Model(inputs, outputs)


# (B) my ramdom-choosen CNN model (~50k para): slightly (deeper) better perf. than (C)
inputs = keras.Input(shape=(28, 28, 1)) # 1 is needed here to keep the same dim with next conv2D layer
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)
x = layers.Dropout(.2)(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)
x = layers.Dropout(.2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x) # <<<<<<<<
x = layers.Dropout(.2)(x)
outputs = layers.Dense(10, activation="softmax")(x)

# (C) a recomended CNN model (~1m para): https://keras.io/examples/mnist_cnn/
# inputs = keras.Input(shape=(28, 28, 1)) # 1 is needed here to keep the same dim with next conv2D layer
# x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
# x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# x = layers.Dropout(.25)(x)
# x = layers.Flatten()(x)
# x = layers.Dense(128, activation="relu")(x)
# x = layers.Dropout(.5)(x)
# outputs = layers.Dense(10, activation="softmax")(x)


model = keras.Model(inputs, outputs)

# show model
model.summary()
elapsed_time = time.time() - start_time
print('elapsed_time:', elapsed_time)

#%% Compile the model
'''
NOTE: in model.compile(), ... tf.keras.metrics=['accuracy'], ... 
the term “accuracy” is an expression, to let the training file decide which 
metric should be used (binary accuracy, categorial accuracy or sparse categorial 
accuracy). This decision is based on certain parameters like the output shape 
(the shape of the tensor that is produced by the layer and that will be the 
input of the next layer) and the loss functions.
'''

## (A) single integers as class lable (Don't convert Y to one-hot class label)
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy", #or, loss= keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
#     )

## (B) one-hot class lable, loss & metrics differ  
model.compile(
    optimizer="adam",
    loss='categorical_crossentropy',   # one-hot class lable 
    metrics= [tf.keras.metrics.CategoricalAccuracy(name ="acc")],# "acc" for picking data from dict
    ) # 'metrics.Accuracy' will yield extremly low acc due to directly compare y and y_hat

callbacks = [keras.callbacks.TensorBoard(log_dir='./logs')] # for using TensorBoard
# Launch TensorBoard from the command line: tensorboard --logdir= path_to_your_logs

#%% train
N_epoch = 50
start_time = time.time()

# (A) Train the model from Numpy data
history = model.fit(x_train, y_train, validation_data= (x_test, y_test), 
                    batch_size=batch_size, epochs=N_epoch, callbacks=callbacks)

# (B)  Train the model using a dataset
# history = model.fit(dataset, epochs=40, callbacks=callbacks)
# history = model.fit(dataset, epochs=N_epoch, validation_data= val_dataset)

print('elapsed_time:',  time.time() - start_time)

# plot training process
hist_dict = history.history
fig = plt.figure
plt.plot(hist_dict['loss'],label='loss',linestyle='-')
plt.plot(hist_dict['acc'],label='acc',linestyle='-')
plt.plot(hist_dict['val_loss'],label='val_loss',linestyle='--')
plt.plot(hist_dict['val_acc'],label='val_acc',linestyle='--')
plt.legend(loc='best', fontsize='x-large')
plt.grid(True)

#%% evaluate & predict
# loss, acc = model.evaluate(val_dataset)  # returns loss and metrics
loss, acc = model.evaluate(x_test, y_test)
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)

# predictions = model.predict(val_dataset)
predictions = model.predict(x_test)
# predictions = model.predict_classes(x_test) # Only available for sequential models
y_pred = np.argmax(predictions, axis=1)

y_incorrects_index = np.nonzero(y_pred!= y_test_integer)[0]
print('the total number of incorrect preditions', len(y_incorrects_index))

#%% pick first N samples
num = len(y_incorrects_index)
images = x_test_orig[y_incorrects_index[:num]]
labels = y_test_integer[y_incorrects_index[:num]]
preds = y_pred[y_incorrects_index[:num]]
num_row = 8
num_col = 10
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
    ax.set_ylabel('Pred: {}'.format(preds[i]))
plt.tight_layout()
plt.show()

 

