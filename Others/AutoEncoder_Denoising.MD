# AutoEncoder for Denoising

In the following section, you will create a noisy version of the Fashion MNIST dataset by applying random noise to each image. You will then train an autoencoder using the noisy image as input, and the original image as the target.

# Architecture of Denosing AutoEncoder

![image](https://github.com/hughiephan/DPL/assets/16631121/dbe0125c-e950-45fe-a0fb-118754cb2024)

The architecture of a denoising autoencoder (DAE) is similar to that of a standard autoencoder. It consists of an encoder, which maps the input data to a lower-dimensional representation, or encoding, and a decoder, which maps the encoding back to the original data space. During training, the autoencoder is given a set of clean input examples `x_train / x_test` and the corresponding noisy versions of these examples `x_train_noisy / x_test_noisy`. The goal is to learn a function that maps the noisy input to the clean output using the encoder-decoder architecture.

## Step 1: Import TensorFlow and other libraries
```py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
```

## Step 2: Train the basic autoencoder using the Fashion MNIST dataset. 
Each image in this dataset is 28x28 pixels. `tf.newaxis` is similar to `expand_dims()` which adds a new axis.  `/ 255` to normalize the pixels into [0, 1] range

```py
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
```

## Step 3: Adding random noise to the images

`tf.random.normal()` function is used to create a tf.Tensor with values sampled from a normal distribution (https://www.quora.com/What-is-the-normal-distribution-Could-you-explain-it-to-me-with-easy-language-and-example). The `clip_by_value()` in TensorFlow clips values of a tensor to a specified minimum and maximum value.
![image](https://github.com/hughiephan/DPL/assets/16631121/4f42e3b5-e774-49a2-bf69-e5de48917ccd)

```py
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)
```

## Step 4: Define a convolutional autoencoder
Train a convolutional autoencoder using Conv2D layers in the encoder, and Conv2DTranspose layers in the decoder. `Conv2D` is mainly used when you want to detect features and `Conv2DTranspose` is used for constructing features (https://stackoverflow.com/questions/68976745/in-keras-what-is-the-difference-between-conv2dtranspose-and-conv2d)
```py
class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(28, 28, 1)),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
```

Encoder downsamples the images from 28x28 to 7x7.

![image](https://github.com/hughiephan/DPL/assets/16631121/1f27edd0-b1f0-49b0-b8d9-37b6e12ea388)

Decoder upsamples the images back from 7x7 to 28x28.
![image](https://github.com/hughiephan/DPL/assets/16631121/c373a000-4096-4a60-bde9-3d6047e34b3f)


## Step 5: Training
```py
autoencoder = Denoise()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))
```

## Step 6: Evaluate
Plotting both the noisy images and the denoised images produced by the autoencoder. `tf.squeeze()` removes dimensions whose size is 1. For example: (28, 28, 1) will become (28, 28), which is similar to (width, height) of an image.

![image](https://github.com/hughiephan/DPL/assets/16631121/8d6a8b83-d2b3-4cc8-83e2-08a9d757b433)

```py
encoded_imgs = autoencoder.encoder(x_test_noisy).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()
```
