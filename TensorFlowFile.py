import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


x_train = tf.constant([3,4,6,9,13], dtype=tf.float32)
y_train = tf.constant([9,16,36,81,169], dtype=tf.float32)

model = keras.Sequential([
    layers.Dense(200, activation='relu', input_shape=(1,)),
    layers.Dense(100),
    layers.Dense(1) ])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(x_train,y_train,epochs=600, verbose=0)
print("Training Completed")

x_test = tf.constant([6, 8, 11, 12, 13], dtype=tf.float32)
predictions = model.predict(x_test)

print("Input data:", x_test)
tf.print("Predictions:", predictions.flatten())

x_train_weight = tf.constant([5,10,15,20,25], dtype=tf.float32)
x_train_distance = tf.constant([50,100,150,200,250], dtype=tf.float32)
y_train_fuel = tf.constant([500,1000,1500,2000,2500], dtype=tf.float32)

x_train = tf.stack([x_train_weight, x_train_distance], axis=1)

model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(2,)),
    layers.Dense(5),
    layers.Dense(1) ])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(x_train,y_train_fuel,epochs=500, verbose=0)
print("Training Completed")

x_test_weight = tf.constant([8, 12, 18], dtype=tf.float32)
x_test_distance = tf.constant([60, 120, 180], dtype=tf.float32)
x_test = tf.stack([x_test_weight, x_test_distance], axis=1)
predictions = model.predict(x_test)
print("Fuel:", predictions.flatten())