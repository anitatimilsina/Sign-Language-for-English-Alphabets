from tqdm import tqdm
from pickle import load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten

x = load(open("features.pkl", "rb"))
y = load(open("labels.pkl", "rb"))

# Normalization of Features
x = x/255.0

print("The shape of the data features is:", x.shape)
print("The shape of the data class labels is:", y.shape)

input_img_shape = x.shape[1:]
print("The shape of the image to be fit to the model is:", input_img_shape)


# Designing Model
model = Sequential()

model.add(Conv2D(16, (2,2), input_shape=input_img_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='same'))

model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(5, 5), strides=(5, 5), padding='same'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
# 26, here is the number of class labels
model.add(Dense(26, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Here we will train the model and save our model after each epoch as .h5 filename begining with model_
epochs = 15
for i in tqdm(range(epochs)):
    model.fit(x, y, batch_size=32, epochs=1, validation_split=0.2, verbose=1)
    model.save(f"./models/model_{i}.h5" )