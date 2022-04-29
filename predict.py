import cv2
import numpy as np
from pickle import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


CLASSES = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

def predict_output(filename):
    img_arr = cv2.imread(filename)
    img_arr = cv2.resize(img_arr, (80, 80))
    img_arr = img_arr/255.0
    print("The shape of the image array before expanding is:", img_arr.shape)
    img_arr = np.expand_dims(img_arr, axis=0)
    print("The shape of the image array before expanding is:", img_arr.shape)
    
    model = load_model('./models/model_12.h5')
    result = np.argmax(model.predict(img_arr), axis=1)
    result = CLASSES[int(result)]
    return result


filename = "./images/a1.jpg"
result = predict_output(filename)

print("The given image of sign language is equivalent to English alphabet:", result)