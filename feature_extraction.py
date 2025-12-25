import os
import numpy as np
import pickle
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = GlobalMaxPooling2D()(base_model.output)
model = Model(base_model.input, x)


image_dir = "images"
image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


    
features = []
filenames = []

for file in tqdm(os.listdir('images')):
    path = os.path.join('images', file)
    try:
        img = load_and_preprocess(path)
        embedding = model.predict(img, verbose=0)
        features.append(embedding.flatten())
        filenames.append(path)
    except:
        continue

features = np.array(features).astype('float32')

pickle.dump(features, open('embeddings.pkl','wb'))
pickle.dump(filenames, open('filenames.pkl','wb'))

np.save("embeddings.npy", features)