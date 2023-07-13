# AutoEncoder for Anomaly Detection

In this example, you will train an autoencoder to detect anomalies on the ECG5000 dataset http://www.timeseriesclassification.com/description.php?Dataset=ECG5000.  This dataset contains 5,000 Electrocardiograms, each with 140 data points. You will use a simplified version of the dataset, where each example has been labeled either 0 (corresponding to an abnormal rhythm), or 1 (corresponding to a normal rhythm). Our goal is to identify abnormal rhythms.

![image](https://github.com/hughiephan/DPL/assets/16631121/97729dba-0a1a-482f-84d7-6e6c95d8ac01)

## Step 1: Import TensorFlow and other libraries
```py
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
```

## Step 2: Download the dataset

```py
dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()
```

## Step 3: Pre-process the data
`tf.reduce_min / tf.reduce_max` operations are to get the minimum and maximum value of a TensorFlow tensor. Then we normalize the data to [0,1] by dividing with `(max_val - min_val)`.It is similar to dividing by 255 for image processing). Then apply `tf.cast` to cast the tensor to float type.

```py
labels = raw_data[:, -1] # Last element contains the labels
data = raw_data[:, 0:-1] # Other data points are the ECG data
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)
train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)
train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)
```

![image](https://github.com/hughiephan/DPL/assets/16631121/7fa9c39c-c22d-4859-9414-6e1aa910d563)

## Step 4: Separate the normal rhythms from the abnormal rhythms.
Train the autoencoder using only the normal rhythms, which are labeled in this dataset as 1
```py
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)
normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]
anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]
```

Results of `train_labels` and `~train_labels`

![image](https://github.com/hughiephan/DPL/assets/16631121/9ebe42a9-a5a3-436d-8b34-3d8883087438)

![image](https://github.com/hughiephan/DPL/assets/16631121/b985c836-1e9d-4266-a361-dd18176245b4)

![image](https://github.com/hughiephan/DPL/assets/16631121/ba8a88cc-0a7d-4249-b1a0-742eac6ed05c)

## Step 5: Build the model
```py
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])
    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
```

## Step 6: Training
```py
autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=20, 
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)
```

## Step 7: Set a threshold
Detect anomalies by calculating whether the reconstruction loss is greater than a fixed threshold. 
In this tutorial, you will calculate the mean average error for normal examples from the training set, then classify future examples as anomalous if the reconstruction error is higher than one standard deviation from the training set (https://www.quora.com/What-is-meant-by-one-standard-deviation-away-from-the-mean). And choose a threshold value that is one standard deviation above the mean

```py
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
threshold = np.mean(train_loss) + np.std(train_loss)
```

Example of one standard deviation away from the mean

![image](https://github.com/hughiephan/DPL/assets/16631121/cc34bb2d-b25f-4476-a4f4-2de819c0fd0b)

## Step 8: Define prediction function
Classify an ECG as an anomaly if the reconstruction error is greater than the threshold (one standard deviation) from the normal training examples. We use Mean Absolute Error `tf.keras.losses.mae` to calculate the difference between reconstruction and our input.

```py
def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)
```

Example of `tf.math.less` function

![image](https://github.com/hughiephan/DPL/assets/16631121/f4dbe95c-3402-4a68-b9ec-336afcdae61d)

## Step 9: Make prediction
```py
predictions = predict(autoencoder, test_data, threshold)
print("Accuracy = {}".format(accuracy_score(test_labels, predictions)))
print("Precision = {}".format(precision_score(test_labels, predictions)))
print("Recall = {}".format(recall_score(test_labels, predictions)))
```

![image](https://github.com/hughiephan/DPL/assets/16631121/6bd1ed9f-7305-4b73-8435-fd19c4abd0c0)

![image](https://github.com/hughiephan/DPL/assets/16631121/fc723dc7-042a-4c8b-a168-8b206639505c)
