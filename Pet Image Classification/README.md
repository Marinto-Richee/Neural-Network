
# <p align="center">Pet Classification</p>
## Algorithm:
1. Import necessary packages.
2. Read the dataset and normalize the data.
3. Form the CNN model using the necessary layers and filters.
4. Train the model using the training dataset. 
5. Evalute the model using the test data.
6. Test the model upon various new images of dogs and cats.

## Program:

```python3
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten                             
```
```python3
X_train=np.loadtxt('input.csv',delimiter=',')
Y_train=np.loadtxt('labels.csv',delimiter=',')
X_test=np.loadtxt('input_test.csv',delimiter=',')
Y_test=np.loadtxt('labels_test.csv',delimiter=',')
X_train=X_train.reshape(len(X_train),100,100,3)
X_test=X_test.reshape(len(X_test),100,100,3)
Y_train=Y_train.reshape(len(Y_train),1)
Y_test=Y_test.reshape(len(Y_test),1)
X_test=X_test/255.0
X_train=X_train/255.0
print("shape of x_train:",X_train.shape)
print("Shape of Y_train:",Y_train.shape)
print("Shape of x_test:",X_test.shape)
print("Shape of Y_test:",Y_test.shape)
```
```python3
model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)),
    MaxPooling2D((2,2)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
])
```
```python3
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
with tf.device("DML"):
    model.fit(X_train,Y_train,epochs=10,batch_size=32)
model.evaluate(X_test,Y_test)
```

## Output:
![image](https://user-images.githubusercontent.com/65499285/226119474-e553fa1f-fe9b-46f5-8ace-114d0ad7d32a.png)
![image](https://user-images.githubusercontent.com/65499285/226119616-a5110d62-ece1-4c2f-9263-bcad59ae9322.png)
![image](https://user-images.githubusercontent.com/65499285/226119482-a0986093-ccc2-4a01-b85d-a23d39fd5788.png)
![image](https://user-images.githubusercontent.com/65499285/226119493-7725e0c2-87f0-421c-af38-7fe22352b499.png)


