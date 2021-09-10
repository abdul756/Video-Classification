from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
train = pd.read_csv('train_new_final.csv')

train_image = []

# for loop to read and store frames
folder_p = ['carcrash','falling','hitting','kicking','neutral','running','sitting','stealing','vandalizing','walking']

# for j in folder_p:
#     print(j)
#     # quit()
for i in tqdm(range(train.shape[0])):
    # folder_p=['falling','carcrash']
    # loading the image and keeping the target size as (224,224,3)
    # for j in range(3) :
    #     print(j)
    #     quit()
        img = image.load_img("train_01/"+train['class'][i] + '/'+ train['image'][i], target_size=(128, 128, 3))
        # converting it to array
        img = image.img_to_array(img)
        # normalizing the pixel value
        img = img / 255
        # appending the image to the train_image list
        train_image.append(img)

# converting the list to numpy array

X = np.array(train_image)
del train_image
# shape of the array
print(X.shape)

y = train['class']

# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)
del X
# creating dummies of target variable for train and validation set
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
# print("________________________",y_train.shape,y_test.shape)
# quit()
width,height=128,128

n_class=2
from keras.models import *
from keras.layers import *
rnn_size = 24

input_tensor = Input((width, height, 3))
x = input_tensor

x = Convolution2D(32, 3, 3, activation='relu')(x)
x = Convolution2D(32, 3, 3, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

# x = Dense(32, activation='relu')(x)

gru_1 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = LSTM(rnn_size, return_sequences=True, go_backwards=False, init='he_normal', name='gru1_b')(x)
gru1_merged = Add()([gru_1, gru_1b])

gru_2 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = LSTM(rnn_size, return_sequences=False, go_backwards=False, init='he_normal', name='gru2_b')(gru1_merged)
x = concatenate([gru_2, gru_2b])
# x = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
x = Dropout(0.25)(x)
x=Flatten()(x)
x = Dense(10, activation='softmax')(x)
model = Model(input=input_tensor, output=x)
# defining a function to save the weights of best model
from keras.callbacks import ModelCheckpoint
mcp_save = ModelCheckpoint('weight_new_all.hdf5', save_best_only=True, monitor='val_loss', mode='min')
import time
log_file_name = 'realtime_activitiy_model_{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(log_file_name))
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001)
# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# training the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[tensorboard,mcp_save,reduce_lr], batch_size=128)
history = model.history.history
def plot_metrics(history):

    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['acc']
    val_acc = history['val_acc']

    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('loss.png')


    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    # plt.show()
    plt.savefig('Acc.png')




plot_metrics(history)
