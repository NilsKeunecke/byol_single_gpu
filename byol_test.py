import os

# Reduce tensorflow log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import sklearn

# Limit memory growth
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

num_classes = 10

# load Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
labels = y_train
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# Define custom layers and regularizers for successful loading
def ws_reg(kernel):
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)
    # return kernel


# load Model
trained_model = tf.keras.models.load_model(filepath='saved_model/test/byol_train_epoch_14_loss_0.00126.hdf5',
                                           custom_objects={'GroupNormalization': tfa.layers.GroupNormalization,
                                                           'ws_reg': ws_reg}, compile=False)

for layer in trained_model.layers:
    layer.trainable = False

x = keras.layers.Dense(4096, activation="relu", name='predictor_1')(trained_model.get_layer("dense_1").output)
x = keras.layers.BatchNormalization(name='bn_bn1')(x)
x = keras.layers.Dense(256, activation="relu", name='predictor_2')(x)
x = keras.layers.BatchNormalization(name='bn_bn2')(x)
x = keras.layers.Flatten()(x)
output = keras.layers.Dense(num_classes, activation="softmax", name='predictor_3')(x)

test_model = Model(inputs=trained_model.input, outputs=trained_model.get_layer("dense_1").output)
model = Model(inputs=trained_model.input, outputs=output)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(learning_rate=3e-4),
              metrics=['accuracy'])


# Look at a random prediction (Many models degrade and show all zeros)
print(test_model(np.reshape(x_train[0], [1, 32, 32, 3])))
print(tf.math.reduce_max(test_model(np.reshape(x_train[0], [1, 32, 32, 3]))))
print(tf.math.reduce_min(test_model(np.reshape(x_train[0], [1, 32, 32, 3]))))


# print tsne graph
print("build TSNE")
colors = {0 : "#00ffff",
          1 : "#aaaaaa",
          2 : "#ffffff",
          3 : "#ff0000",
          4 : "#00ff00",
          5 : "#0000ff",
          6 : "#ffff00",
          7 : "#ff00ff",
          8 : "#0f0f0f",
          9 : "#f0f0f0"
}

intermediates = []
color_intermediates = []
x_train = np.reshape(x_train, [50000, 1, 32, 32, 3])
for i in range(1000):
    output_class = int(labels[i])
    intermediate_tensor = test_model.predict(x_train[i])
    intermediates.append(intermediate_tensor[0])
    color_intermediates.append(colors[output_class])

print("compute graph")
from sklearn.manifold import TSNE
tsne = TSNE(n_components=3, random_state=0)
intermediates_tsne = tsne.fit_transform(intermediates)

plt.figure(figsize=(8, 8))
plt.scatter(x=intermediates_tsne[:,0], y=intermediates_tsne[:,1], color=color_intermediates)
plt.show()
exit()

""" Fine-tune on x percent script. Not used yet."""
'''
# Adjust Dataset
x_train_one = []
y_train_one = []

for label in range(10):
    num_samples = 0
    for idx in range(50000):
        if num_samples == 50:
            break
        if y_train[idx].astype(dtype=int) == label:
            num_samples += 1
            x_train_one.append(x_train[idx])
            y_train_one.append(y_train[idx])

# x_train_one = x_train_one.astype('float32') / 255.0
x_train_one = np.reshape(x_train_one, (500, 32, 32, 3))
y_train_one = to_categorical(y_train_one, num_classes)

print(np.shape(x_train_one))
print(np.shape(y_train_one))
'''

# Fine-Tune
model.fit(x_train, y_train, batch_size=64, epochs=30, verbose=1, shuffle=True)
# Evaluate on the test data
model.evaluate(x_test, y_test, batch_size=64, verbose=1)
