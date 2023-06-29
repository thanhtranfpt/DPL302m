# Transfer Learning with MobileNet + SVM 
Transfer learning is used for feature extraction. Pretrained MobileNet is loaded from Keras applications and used to get a feature vector for each image in the dataset. Then we use SVM for classifying.

## Step 1: Import libraries
```py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
```

## Step 2: Import Dataset
Flowers Recognition dataset https://www.kaggle.com/datasets/alxmamaev/flowers-recognition will be used. In this dataset, there are 5 classes: daisy, dandelion, rose, sunflower, and tulip. Images belonging to a specific class are collected in the corresponding folder name
```py
classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
path = "/kaggle/input/flowers-recognition/flowers"
```
![image](https://github.com/hughiephan/DPL/assets/16631121/d80667e4-4aa5-4ad9-90fc-4fa6db7a2e38)

## Step 3: Import Model
MobileNet is an efficient and portable CNN architecture. Because of the small size of the model, these models are considered very useful to be implemented on mobile and embedded devices. While a VGG16 model can take up to 500 MB of disk space, MobileNet just needs 16–18MB. This makes it ideal to be loaded on mobile devices. Having `include_top=True` means that a fully-connected layer will be added at the end of the model

```py
model = MobileNet(input_shape=(224,224,3), include_top=True)
model.summary()
```
![image](https://github.com/hughiephan/DPL/assets/16631121/69cb2019-a45f-4a36-aa85-dbaedcd01d08)

## Step 4: Feature Extractor
The output of reshape_2 layer is a 1000 elements vector. MobileNet predictions layer takes this vector as input. We use this vector as our feature and try to do classification. A new model `feature_extractor` is created.

```py
vector = model.get_layer("reshape_2").output
feature_extractor = tf.keras.Model(model.input, vector)
```

## Step 5: Run Feature Extract
All files in 5 folders are scanned one by one. There are some non-image files in dataset folders. Extension of each file is checked and file is skipped if its extension is not jpg. Images are read, resized, converted to numpy array and preprocessed. Then, feature_extractor takes image and produces corresponding feature vector. 

```py
X = []
Y = []
for f in range(5): # Flowers folder have 5 classes
    folder_path = os.path.join(path, classes[f])
    for file in os.listdir(folder_path):    
        file_path = os.path.join(folder_path, file)
        if not(file.endswith(".jpg")):
            continue
        img = image.load_img(file_path, target_size=(224,224))
        img_arr = image.img_to_array(img)
        img_arr_b = np.expand_dims(img_arr, axis=0)
        input_img = preprocess_input(img_arr_b)
        feature_vec = feature_extractor.predict(input_img)
        X.append(feature_vec.ravel())
        Y.append(f)
```
![image](https://github.com/hughiephan/DPL/assets/16631121/94cbceba-2e7a-4f98-a072-90f5c2008a01)

Looking at the first layer of `model.summary()` you can see it's expecting data in this format `[(None, 224, 224, 3)]` so we need to add 1 more dimension by `np.expand_dims(img_arr, axis=0)`, after that the image will transform from `(224, 224, 3)` into `(1, 224, 224, 3)`.

![image](https://github.com/hughiephan/DPL/assets/16631121/ef2c9233-602b-4c73-a57f-43c869d771cf)

## Step 6: Train Test Split
`X` and `Y` Lists are converted to numpy arrays. Then further split them into train and test sets using `train_test_split`. We want stratified splits with same class distributions.
```py
X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42, shuffle=True) # Read more about Stratify here https://dragonforest.in/stratify/
```
![image](https://github.com/hughiephan/DPL/assets/16631121/ae440bdc-2be2-4b9d-b992-2117ee82a170)

## Step 7: Make prediction with SVM
```py
svm_model = svm.SVC(C=1.0, kernel="rbf")
svm_model.fit(train_X, train_Y)
y_pred = svm_model.predict(test_X)
print(classification_report(test_Y, y_pred,target_names=classes))
```
![image](https://github.com/hughiephan/DPL/assets/16631121/adaa6b93-7a3b-428e-8167-ce224e2938e3)
