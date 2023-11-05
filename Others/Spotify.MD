# References
EDA
- https://www.kaggle.com/code/muhammedtausif/top-songs-eda

Original Dataset
- https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify

Step 1: Import necessary libraries
```py
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from tensorflow import keras
warnings.simplefilter("ignore")
```
Step 2: Load Dataset
```py
data = pd.read_csv("/kaggle/input/genres-v2-spotify/genres_v2.csv") # Download dataset from 
data
```
You can use this dataset https://www.kaggle.com/datasets/thala321/genres-v2-spotify . Then we load the dataset using pandas read_csv, and the data set contains 42305 rows and 22 columns and consists of 18000+ tracks. 
![Screenshot_2023-03-01_at_9](https://github.com/hughiephan/DPL/assets/16631121/bc6618d7-bc0c-476b-a066-576c0a9f2e1a)


Step 3: Exploring the Data
```py
data.iloc[:,:20]
```
We can use the ‘iloc’ method to select the rows and columns that form a data frame by their integer index positions. Example of choosing the first 20 columns of the df.
![image](https://github.com/hughiephan/DPL/assets/16631121/986a9a9f-871c-407e-9bd4-5806a0a826f7)
```py
data.info()
```
When you call data.info(), it will print the following information:
- The number of rows and columns in the data frame.
- The name of each column, its data type, and the number of non-null values in that column.
- The total number of non-null values in the data frame.
- The memory usage of the DataFrame.

![image](https://github.com/hughiephan/DPL/assets/16631121/bd15ecd5-1417-4ab3-9fb1-2b573f17cffd)

Step 4: Data Cleaning 
```py
df = data.drop(["type","type","id","uri","track_href","analysis_url","song_name", "Unnamed: 0","title", "duration_ms", "time_signature"], axis =1) 
df
```
Here, we want to clean our data by removing unnecessary columns that add no value to the prediction. We have removed some columns that add no value to this particular problem and put axis = 1, where it drops the columns rather than rows. If you specify axis=1 you will be removing columns. If you specify axis=0 you will be removing rows from dataset. We are again calling the Data Frame to see the new Data Frame with helpful information.
![image](https://github.com/hughiephan/DPL/assets/16631121/c9b0ccfa-b056-45ec-af45-70e8e3429f07)

Step 5: Describe the data
```py
df.describe()
```
The df.describe( ) method generates descriptive statistics of a Pandas Data Frame. It provides a summary of the central tendency and dispersion and the shape of the distribution of a dataset.
After running this command, you can see all the descriptive statistics of the Data Frame, like std, mean, median, percentile, min, and max.
![image](https://github.com/hughiephan/DPL/assets/16631121/175db13c-d661-4737-8139-43eac2c8ebfe)

```py
df["genre"].value_counts()
```

Step 6: Genre Types
```py
ax = sns.histplot(df["genre"]) 
_ = plt.xticks(rotation=60) 
_ = plt.title("Genres")
```
axe = sns.histplot(df[“genre”]) generates a histogram of the distribution of values in a Pandas DataFrame named df’s “genre” column. This code may be used to visualize the frequency of some Spotify genres in a music dataset.

![image](https://github.com/hughiephan/DPL/assets/16631121/6c2d6189-9757-4cb6-acb9-ce06c182e11c)

Step 7: Drop and show Data Correlation
```py
df.corr()
```
We computes the correlation matrix of the DataFrame to find correlations between the features.

![image](https://github.com/hughiephan/DPL/assets/16631121/7b0261af-821f-4dd4-9804-59bc5a8beb77)

Step 8: Define Feature Input X and Target Output Y

```py
x = df.drop('genre', axis=1)
y = df["genre"]
```
We remove the Genre column from the feature data so our model can learn to fit X and predict y

```py
x
```
![image](https://github.com/hughiephan/DPL/assets/16631121/9b71b69d-9131-4702-851d-38656eb12098)

```py
y
```
![image](https://github.com/hughiephan/DPL/assets/16631121/9e22c5d7-e364-4ec1-a2ab-c3aebe7428b9)

```py
y.unique()
```
![image](https://github.com/hughiephan/DPL/assets/16631121/aace972d-e1a0-43f7-a7e1-8ad7a6e361dc)


Step 9: Plot data distribution
```py
k=0 
plt.figure(figsize = (18,14)) 
for i in x.columns: 
  plt.subplot(4,4, k + 1) 
  sns.distplot(x[i]) 
  plt.xlabel(i, fontsize=11) 
  k +=1
```

The given code generates a grid of distribution plots that allow users to view the distribution of values over several columns in a dataset. Discovering patterns, trends, and outliers in the data by showing the distribution of values in each column. These are helpful and beneficial for exploratory data analysis and finding valuable and potential faults or inaccuracies in a dataset.

![image](https://github.com/hughiephan/DPL/assets/16631121/27f87946-ce49-4fa3-bb33-9f657a4b591f)


Step 10: Split data
```py
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size= 0.2, random_state=42, shuffle = True)
```
Above code divide the input data into training and testing sets using the Sklearn library’s train test split function

```py
scalerx = MinMaxScaler()
xtrain = scalerx.fit_transform(xtrain)
xtest = scalerx.transform(xtest)
```
Then MinMaxScaler is used for scaling and normalizing the data. The purpose of using MinMaxScaler is to ensure that all the features are on a similar scale, preventing certain features from dominating the learning algorithm due to their larger magnitude. Then we use `fit_transform` for train data and `transform` for test data. `fit_transform` means to do some calculation and then do transformation. So for training set, you need to both calculate and do transformation. But for testing set, Machine learning applies prediction based on what was learned during the training set and so it doesn't need to calculate, it just performs the transformation.

```py
le = preprocessing.LabelEncoder()
ytrain = le.fit_transform(ytrain)
ytest = le.transform(ytest)
print(dict(zip(le.classes_, le.transform(le.classes_))))
```
The LabelEncoder() function from the sklearn.preprocessing package is used to encode labels. It uses the fit_transform() and transform() routines to encode the category target variables (ytrain and ytest) into numerical values. You can read more about LabelEncoder here https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

![image](https://github.com/hughiephan/DPL/assets/16631121/857926b4-4f75-48b3-8b58-0850cd66e5f8)

Step 11: Define early stopping 
```py
early_stopping1 = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 10, restore_best_weights = True) 
early_stopping2 = keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 10, restore_best_weights = True) 
```
This code creates two early stopping callbacks for model training, one based on validation loss and the other on validation accuracy. `Patience = 10` means we will stop if there is no improvement in the last 10 epochs. `restore_best_weights = False` will return the final weights.

Step 12: Define our Neural Network
```py
model = keras.Sequential([ 
  keras.layers.Input(name = "input", shape = (xtrain.shape[1])),  
  keras.layers.Dense(256, activation = "relu"),
  keras.layers.BatchNormalization(), 
  keras.layers.Dropout(0.2), 
  keras.layers.Dense(128, activation = "relu"),   
  keras.layers.BatchNormalization(), 
  keras.layers.Dropout(0.2), 
  keras.layers.Dense(64, activation = "relu"), 
  keras.layers.Dense(max(ytrain)+1, activation = "softmax") 
]) 
model.summary()
```
Keras’ Sequential API makes a NN model with various connected layers using the ReLU activation function, batch normalization, and dropout regularisation. `Dense` is the connected layer that connects each unit of the layer input with each output unit of this layer. `xtrain.shape[1]` will print the number of features, while `max(ytrain)+1` equals to the number of the output target (Genres) that we want to predict. `BatchNormalization` means we use the mini-batch statistics (mean and variance) for normalization rather than population statistics (computed over the entire training dataset) which lead to faster convergence, better stability, and improved generalization. And in the final output layer, we can outputs class probabilities using the softmax activation function (https://www.analyticsvidhya.com/blog/2021/04/introduction-to-softmax-for-neural-network/)

![image](https://github.com/hughiephan/DPL/assets/16631121/42b02a59-cd7c-4090-b951-158f6ac2c621)

Step 13: Train the Neural Network
```py
model.compile(optimizer = keras.optimizers.Adam(), 
             loss = "sparse_categorical_crossentropy", 
             metrics = ["accuracy"])
 
model_history = model.fit(xtrain, ytrain, 
               epochs = 100, 
               verbose = 1, batch_size = 128, 
               validation_data = (xtest, ytest), 
               callbacks = [early_stopping1, early_stopping2])
```
The following code block will compile and train a neural network model. `sparse_categorical_crossentropy` is a loss function commonly used in classification tasks where the target labels are integers. If the target labels are one-hot encoded vectors, the regular `categorical cross-entropy` loss function should be used. Then we use “Adam” as our optimizer. The model is trained for 100 epochs, batch_size = 128, with callbacks that end early based on validation loss and accuracy.

![image](https://github.com/hughiephan/DPL/assets/16631121/bef5bf14-d8fc-4e4a-a7f7-785a32a74ae5)

Step 14: Evaluate
```py
print(model.evaluate(xtrain, ytrain)) 
print(model.evaluate(xtest, ytest))
```
The training data is xtrain and ytrain, whereas the validation data is sent as xtest and ytest. The training history of the model is saved in the model history variable.

![image](https://github.com/hughiephan/DPL/assets/16631121/16a6ea7d-b149-4c0e-aec3-dc57ddd47866)

Step 15: Plot the results
```py
plt.plot(model_history.history["loss"]) 
plt.plot(model_history.history["val_loss"]) 
plt.legend(["loss", "validation loss"], loc ="upper right") plt.title("Train and Validation Loss") 
plt.xlabel("epoch") 
plt.ylabel("Sparse Categorical Cross Entropy") 
plt.show()
```

![image](https://github.com/hughiephan/DPL/assets/16631121/11f7cc68-3bbd-427a-8e72-9a55518a05d3)

The following code generates a plot using matplotlib; on the x_axis, we have the epoch, and on the y_axis, we have the sparse Categorical Cross Entropy.
```py
plt.plot(model_history.history["accuracy"]) plt.plot(model_history.history["val_accuracy"]) plt.legend(["accuracy", "validation accuracy"], loc ="upper right") plt.title("Train and Validation Accuracy") 
plt.xlabel("epoch") 
plt.ylabel("Accuracy") 
plt.show()
```

![image](https://github.com/hughiephan/DPL/assets/16631121/2f5880e4-dc14-4647-84e2-f8d537d2b135)


Step 16: Run some Prediction
```py
numToPredict = 4
predictMe = df.drop('genre', axis = 1).iloc[0:numToPredict]
print(predictMe)
ypred = model.predict(scalerx.transform(predictMe)).argmax(axis=1)
print('Prediction in Numerical ', ypred)
print('Prediction in Text ', le.inverse_transform(ypred))
print('Actual Value ', df.iloc[0:numToPredict]['genre'].to_numpy())
```
Let's get the first 4 data and predict their genres. We need to `transform` the data using the same scaler we used in the training phase. After we get the prediction, we use `inverse_transform` to get back the Genres in text format instead of Encoded Numerical Label. Finally, compare the Predicted Genres with the Actual Genres.

![image](https://github.com/hughiephan/DPL/assets/16631121/d31740f4-d997-42be-95c4-6f5c55c04417)

Step 17: Save the model
```py
model.save("saved_model.h5")
loaded_model = keras.models.load_model("saved_model.h5")
```

After training, the model.save() method is called to save the model to a file named "saved_model.h5". The ".h5" extension indicates that the model will be saved in the HDF5 format. You can replace "saved_model.h5" with the desired file path where you want to save your model. After saving the model, you can later load it using `keras.models.load_model()` to reuse it for predicting or further training.

