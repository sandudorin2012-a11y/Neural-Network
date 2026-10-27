import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import random

model = tf.keras.models.load_model('fashion_mnist_model.h5')

(_, _), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test / 255
x_test = x_test.reshape(-1, 28 * 28)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

def predict_image(index):
    img = x_test[index]
    img_expanded = img.reshape(None,28,28)

    single_image = np.expand_dims(img, axis=0)
    predictions = model.predict(img_expanded)
    predicted_class = predictions[0].argmax()
    plt.imshow(img, cmap="gray")
    plt.title(f"Preditcted: {class_names[predicted_class]}\n True class: {class_names[y_test[index]]}")


    plt.axis("off")
    plt.show()


random_index = random.randint(0, len(x_test) - 1)
predict_image(random_index)