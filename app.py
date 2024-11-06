# import tensorflow
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
# import numpy as np
# from numpy.linalg import norm
# import os
# from tqdm import tqdm
# import pickle

# model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
# model.trainable = False

# model = tensorflow.keras.Sequential([
#     model,
#     GlobalMaxPooling2D()
# ])

# #print(model.summary())

# def extract_features(img_path,model):
#     img = image.load_img(img_path,target_size=(224,224))
#     img_array = image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_array)
#     result = model.predict(preprocessed_img).flatten()
#     normalized_result = result / norm(result)

#     return normalized_result

# filenames = []

# for file in os.listdir('images'):
#     filenames.append(os.path.join('images',file))

# feature_list = []

# for file in tqdm(filenames):
#     feature_list.append(extract_features(file,model))

# pickle.dump(feature_list,open('embeddings.pkl','wb'))
# pickle.dump(filenames,open('filenames.pkl','wb'))

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from PIL import Image

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add GlobalMaxPooling2D layer to the model
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to validate image files
def is_valid_image(img_path):
    try:
        img = Image.open(img_path)
        img.verify()  # Verify if image is valid
        return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid image file: {img_path}, Error: {e}")
        return False

# Function to extract features from images using the model
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Get the list of image files in the "images" directory
filenames = []

# Only process files with image extensions and skip hidden/system files
for file in os.listdir('images'):
    if not file.startswith('.') and file.endswith(('.jpg', '.jpeg', '.png')):
        filenames.append(os.path.join('images', file))

# Initialize an empty list to store the feature vectors
feature_list = []

# Loop through the image files and extract features
for file in tqdm(filenames):
    if is_valid_image(file):  # Check if the image is valid
        try:
            # Extract features and append to the feature list
            feature = extract_features(file, model)
            feature_list.append(feature)
        except Exception as e:
            # Print error and continue processing other images
            print(f"Error processing {file}: {e}")

# Save the feature list and filenames using pickle
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
