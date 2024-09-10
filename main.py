import os
import requests
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input



def create_folder(car_id):
    # Define the folder name
    folder_name = f"{car_id}"
    # Define the path where the folder should be created
    folder_path = os.path.join('image_folders', folder_name)
    
    # Create the folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    return folder_path

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {save_path}")
    except requests.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")

def process_row(row):
    row_index, row_data = row  # row is a tuple (index, pd.Series)
    car_id = row_data['Lot number']
    image_data_url = row_data['Image URL']
    
    if image_data_url.startswith('http'):
        print(f"Processing row {row_index}, image_data_url: {image_data_url}")
        try:
            response = requests.get(image_data_url, timeout=10)
            response.raise_for_status()
            lot_images = response.json().get('lotImages', [])

            # Create folder for this car
            folder_path = create_folder(car_id)

            # Iterate over all images
            for image in lot_images:
                for link in image['link']:
                    if link['isHdImage']:  # Only download full-size images
                        image_url = link['url'].strip()
                        image_name = f"{image['sequence']}_full.jpg"
                        save_path = os.path.join(folder_path, image_name)
                        
                        # Download the image
                        download_image(image_url, save_path)

            # Process images in the folder
            results = process_images(folder_path)
            # evaluate_damage(results)
            
        except requests.RequestException as e:
            print(f"Error processing row {row_index}. Error: {e}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for row {row_index}. Error: {e}")

# Model Functions
def preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image
    img_array = preprocess_input(img_array)
    return img_array

def predict_damage(img_path):
    img_array = preprocess_image(img_path)
    # Get model predictions
    predictions = model.predict(img_array)
    # Decode the predictions to human-readable labels
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions
def process_images(image_folder):
    results = []
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        print(f"Processing {img_path}")
        predictions = predict_damage(img_path)
        results.append(predictions)
    return results

def evaluate_damage(image_folder_path):
    car_id = os.path.basename(image_folder_path)
    predictions = []
    
    for image_file in os.listdir(image_folder_path):
        if image_file.endswith('_full.jpg'):
            image_path = os.path.join(image_folder_path, image_file)
            print(f"Processing {image_path}")

            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)  # Preprocess the image for ResNet50

            prediction = model.predict(img_array)
            damage_score = np.max(prediction)  # Use the model's output as needed
            predictions.append(damage_score)

    if predictions:
        average_damage_score = np.mean(predictions)
        print(f"Average damage score for images in {image_folder_path}: {average_damage_score}")
        return {"Car ID": car_id, "Average Damage Score": average_damage_score}
    else:
        print(f"No predictions for images in {image_folder_path}")
        return {"Car ID": car_id, "Average Damage Score": None}


def save_results_to_csv(results, output_file='damage_scores.csv'):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# ////////

model = ResNet50(weights='imagenet')

file_path = 'D:\\Automobili\\salesdata (35).csv'
output_folder = 'D:\\Automobili\\images'


columns_of_interest = ['Year', 'Odometer',  ]

interestValue = [2022,3000]
mustBe = ['Est. Retail Value', 'Buy-It-Now Price']


df = pd.read_csv(file_path)
# filtered_df = df[(df['Buy-It-Now Price'] > 0) & (df['Year'] > 2022) & (df['Year'] > 2022)]
vehicle_types_of_interest = [ 'A']
# vehicle_types_of_interest = ['C', 'L', 'A', 'R', 'E']

# Filtriraj DataFrame
# filtered_df = df[df['Vehicle Type'].isin(vehicle_types_of_interest)]
filtered_df = df.head(50)

# unique_title_types = df['Vehicle Type'].unique()
print(filtered_df)
# if all(col in df.columns for col in columns_of_interest):
    # filtered_df = df[
    #     (df['Buy-It-Now Price'] > 0) & 
    #     (df['Year'] == interestValue[0]) & 
    #     (df['Odometer'] < interestValue[1]) & 
    #     (df[mustBe[0]] > df[mustBe[1]])
    # ]

    # Using ThreadPoolExecutor to speed up processing
   
with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_row, filtered_df.iterrows()))

# /////Procena Modela----------------------------------
results = []
image_folders = [f.path for f in os.scandir('image_folders') if f.is_dir()]

for folder in image_folders:
    result = evaluate_damage(folder)
    results.append(result)

# Save all results to a CSV file
save_results_to_csv(results)

# -------------------------------------------------------------------------------

# import os
# import requests
# import json
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# # Initialize the model globally
# model = ResNet50(weights='imagenet')

# def create_folder(car_id):
#     folder_path = os.path.join('image_folders', car_id)
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     return folder_path

# def download_image(url, save_path):
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         with open(save_path, 'wb') as file:
#             file.write(response.content)
#         print(f"Downloaded: {save_path}")
#     except requests.RequestException as e:
#         print(f"Failed to download {url}. Error: {e}")

# def process_row(row):
#     row_index, row_data = row
#     car_id = row_data['Lot number']
#     image_data_url = row_data['Image URL']
    
#     if image_data_url.startswith('http'):
#         print(f"Processing row {row_index}, image_data_url: {image_data_url}")
#         try:
#             response = requests.get(image_data_url, timeout=10)
#             response.raise_for_status()
#             lot_images = response.json().get('lotImages', [])

#             folder_path = create_folder(car_id)

#             for image in lot_images:
#                 for link in image['link']:
#                     if link['isHdImage']:
#                         image_url = link['url'].strip()
#                         image_name = f"{image['sequence']}_full.jpg"
#                         save_path = os.path.join(folder_path, image_name)
#                         download_image(image_url, save_path)

#         except requests.RequestException as e:
#             print(f"Error processing row {row_index}. Error: {e}")
#         except json.JSONDecodeError as e:
#             print(f"Error parsing JSON for row {row_index}. Error: {e}")

# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array

# def predict_damage(img_array):
#     return model.predict(img_array)

# def process_images(image_folder):
#     results = []
#     for img_file in os.listdir(image_folder):
#         img_path = os.path.join(image_folder, img_file)
#         print(f"Processing {img_path}")
#         img_array = preprocess_image(img_path)
#         predictions = predict_damage(img_array)
#         results.append(predictions)
#     return results

# def evaluate_damage(image_folder_path):
#     car_id = os.path.basename(image_folder_path)
#     predictions = []
    
#     for image_file in os.listdir(image_folder_path):
#         if image_file.endswith('_full.jpg'):
#             image_path = os.path.join(image_folder_path, image_file)
#             img_array = preprocess_image(image_path)
#             prediction = predict_damage(img_array)
#             damage_score = np.max(prediction)
#             predictions.append(damage_score)

#     if predictions:
#         average_damage_score = np.mean(predictions)
#         print(f"Average damage score for images in {image_folder_path}: {average_damage_score}")
#         return {"Car ID": car_id, "Average Damage Score": average_damage_score}
#     else:
#         print(f"No predictions for images in {image_folder_path}")
#         return {"Car ID": car_id, "Average Damage Score": None}

# def save_results_to_csv(results, output_file='damage_scores.csv'):
#     df = pd.DataFrame(results)
#     df.to_csv(output_file, index=False)
#     print(f"Results saved to {output_file}")

# # Main Execution
# file_path = 'D:\\Automobili\\salesdata (35).csv'
# output_folder = 'D:\\Automobili\\images'

# df = pd.read_csv(file_path)

# vehicle_types_of_interest = ['A']
# filtered_df = df[df['Vehicle Type'].isin(vehicle_types_of_interest)]

# # Process rows and download images
# with ThreadPoolExecutor(max_workers=10) as executor:
#     executor.map(process_row, filtered_df.iterrows())

# # Evaluate damage for each folder
# results = []
# image_folders = [f.path for f in os.scandir('image_folders') if f.is_dir()]

# for folder in image_folders:
#     result = evaluate_damage(folder)
#     results.append(result)

# # Save all results to a CSV file
# save_results_to_csv(results)



