import pytesseract
from PIL import Image
import cv2
from datetime import datetime
import os
import re
import zipfile
import io
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Cert",
    layout = 'wide',
)

st.title('Cert Scraper SSEF')

def crop_image(img):
    # Get the dimensions of the original image
    height, width, channels = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)

    # Calculate the coordinates for cropping crop1, crop2
    top = int(height / 5.70731707317)
    bottom = int(height / 1.56)
    left_crop1 = int(width / 3.63555555556)
    right_crop1 = int(width / 1.3088)

    top_crop2 = int(height / 1.33714285714)
    bottom_crop2 = int(height / 1.21714285714)
    left_crop2 = int(width / 16.36)
    right_crop2 = int(width / 2)

    # Crop the two parts
    crop1 = img[top:bottom, left_crop1:right_crop1]
    crop2 = img[top_crop2:bottom_crop2, left_crop2:right_crop2]

    return crop1, crop2

# Initialize the Tesseract OCR
def initialize_tesseract():
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Initialize Tesseract
initialize_tesseract()

def process_cropped_images(img):
    # Perform OCR on the cropped image
    extracted_text = pytesseract.image_to_string(img)

    return extracted_text

def extract_gemstone_info(img):
    # Assuming you have the functions crop_image and process_cropped_images defined elsewhere
    crop1, crop2 = crop_image(img)
    extracted_texts = process_cropped_images(crop1)

    extracted_texts = extracted_texts.replace('‘', "'")
    lines = [line for line in extracted_texts.split('\n') if line.strip()]
    if len(lines) == 13:
        df = pd.DataFrame({
            'certNo': [lines[0]],
            'Carat': [lines[1]],
            'SC': [lines[2]],
            'Dimensions': [lines[3]],
            'Color': [lines[4]],
            'Identification': [lines[5]],
            'del': [lines[6]],
            'del1': [lines[7]],
            'del2': [lines[8]],
            'indications': [lines[9]],
            'Origin': [lines[10]],
            'Color0': [lines[12]]
        })
        df['certNo'] = df['certNo'].str.strip('No.')
        df['Dimensions'] = df['Dimensions'].str.replace('mm', '')
        df['Color'] = df['Color'].str.replace('of strong saturation', '').str.replace('of medium saturation', '')
        df['Origin'] = df['Origin'].str.extract(r'Origin:\s(\w+)', expand=False)
        df['Color0'] = df['Color0'].str.extract(r"'(.+)' based on SSEF reference standards", expand=False)
        split_values = df['SC'].str.split(',', expand=True)
        df['Shape'] = split_values[0]
        df['Cut'] = split_values[1]
    else:
        df = pd.DataFrame({
            'certNo': [lines[0]],
            'Carat': [lines[1]],
            'SC': [lines[2]],
            'Dimensions': [lines[3]],
            'Color': [lines[4]],
            'Identification': [lines[5]],
            'del': [lines[6]],
            'del1': [lines[7]],
            'del2': [lines[8]],
            'indications': [lines[9]],
            'Origin': [lines[10]]
        })
        df['certNo'] = df['certNo'].str.strip('No.')
        df['Dimensions'] = df['Dimensions'].str.replace('mm', '')
        df['Color'] = df['Color'].str.replace('of strong saturation', '').str.replace('of medium saturation', '')
        df['Origin'] = df['Origin'].str.extract(r'Origin:\s(\w+)', expand=False)
        split_values = df['SC'].str.split(',', expand=True)
        df['Shape'] = split_values[0]
        df['Cut'] = split_values[1]

    extracted_texts1 = process_cropped_images(crop2)
    lines1 = [line for line in extracted_texts1.split('\n') if line.strip()]
    extracted_dates = []
    pattern = r'\d{1,2} [A-Za-z]+ \d{4}'
    match = re.search(pattern, lines1[1])
    if match:
        extracted_date = match.group(0)
        extracted_dates.append(extracted_date)
    else:
        extracted_dates.append(None)  # Use None for lines without a date

    # Create a DataFrame with the extracted dates
    df1 = pd.DataFrame({'issuedDate': extracted_dates})

    final_df = pd.concat([df, df1], axis=1)
    
    return final_df

def detect_color(text):
    text = str(text).lower()  # Convert the text to lowercase
    if "pigeon blood red" in text:
        return "PigeonsBlood"
    elif "royal blue"  in text:
        return "RoyalBlue"
    else:
        return text
    
def detect_cut(cut):
    text = str(cut).lower()
    if "sugar loaf" in text :
        return "sugar loaf"
    elif "cabochon" in text:
        return "cab"
    else:
        return "cut"
    
def detect_shape(shape):
    valid_shapes = [
        "cushion", "heart", "marquise", "octagonal", "oval",
        "pear", "rectangular", "round", "square", "triangular",
        "star", "sugarloaf", "tumbled"
    ]
    if shape in valid_shapes:
        return shape
    else:
        return "Others"
    
def detect_origin(origin):
    if not origin.strip():
        return "No origin"
    
    # Remove words in parentheses
    origin_without_parentheses = origin
    return origin_without_parentheses.strip()

def reformat_issued_date(issued_date):
    try:
        # Remove ordinal suffixes (e.g., "th", "nd", "rd")
        cleaned_date = re.sub(r'(?<=\d)(st|nd|rd|th)\b', '', issued_date.replace("‘", "").strip())

        # Parse the cleaned date string
        parsed_date = datetime.strptime(cleaned_date, '%d %B %Y')

        # Reformat the date to YYYY-MM-DD
        reformatted_date = parsed_date.strftime('%Y-%m-%d')
        return reformatted_date
    except ValueError:
        return ""
    
def detect_mogok(origin):
    return "(Mogok, Myanmar)" in origin


def generate_indication(comment):
    if comment in ["H", "H(a)", "H(b)", "H(c)"]:
        return "Heated"
    else:
        return "Unheated"
    
def detect_old_heat(comment, indication):
    if indication == "Heated":
        if comment in ["H", "H(a)", "H(b)", "H(c)"]:
            return "oldHeat"
    return ""

def generate_display_name(color, Color_1, origin, indication, comment):
    display_name = ""

    if color is not None:
        color = str(color).lower()  # Convert color to lowercase
        if indication == "Unheated":
            display_name = f"SSEF({Color_1})"
        if indication == "Heated": 
            display_name = f"SSEF({Color_1})(H)"
    
    if "(mogok, myanmar)" in str(origin).lower():  # Convert origin to lowercase for case-insensitive comparison
        display_name = "MG-" + display_name
    
    return display_name

# Define the function to extract the year and number from certNO
def extract_cert_info(df,certName):
    # Split the specified column into two columns
    df['certName'] = 'SSEF'
    df['certNO'] = df[certName]
    return df

def convert_carat_to_numeric(value_with_unit):
    value_without_unit = value_with_unit.replace(" ct", "").replace(" et", "").replace(" ot", "")
    numeric_value = (value_without_unit)
    return numeric_value

def convert_dimension(dimension_str):
    parts = dimension_str.replace("—_", "").replace("_", "").replace("§", "5").replace(",", ".").replace("=", "").split(" x ")
    length = (parts[0])
    width = (parts[1])
    height = (parts[2])
    return length, width, height

def rename_identification_to_stone(dataframe):
    # Rename "Identification" to "Stone"
    dataframe.rename(columns={"Identification": "Stone"}, inplace=True)
    
    # Remove unwanted words and trim spaces in the "Stone" column
    dataframe["Stone"] = dataframe["Stone"].str.replace("‘", "").str.strip()

    # Define a list of gemstone names to detect
    gemstone_names = ["Ruby", "corundum", "Emerald", "Pink Sapphire", "Purple Sapphire", "Sapphire", "Spinel", "Tsavorite", "Blue Sapphire", "Fancy Sapphire", "Peridot", "Padparadscha"]  # Add more gemstone names as needed

    # Function to remove "Natural" or "Star" from the stone name
    def remove_prefix(name):
        for prefix in ["Natural", "Star"]:
            name = name.replace(prefix, "").strip()
        return name

    # Detect and update the "Stone" column with the gemstone names (ignoring "Natural" or "Star")
    dataframe["Stone"] = dataframe["Stone"].apply(lambda x: next((gem for gem in gemstone_names if gem in remove_prefix(x)), x))

    return dataframe

# Define the function to perform all data processing steps
def perform_data_processing(img):
    
    result_df = extract_gemstone_info(img)
    
    result_df["Detected_Origin"] = result_df["Origin"].apply(detect_origin)
    result_df["Indication"] = result_df["indications"].apply(generate_indication)
    result_df["oldHeat"] = result_df.apply(lambda row: detect_old_heat(row["indications"], row["Indication"]), axis=1)
    
    if "Color0" in result_df and "Color" in result_df:
        result_df["Detected_Color"] = result_df.apply(lambda row: detect_color(row["Color0"]), axis=1)
        result_df["displayName"] = result_df.apply(lambda row: generate_display_name(row["Color0"], row['Detected_Color'], row["Detected_Origin"], row['Indication'], row['oldHeat']), axis=1)
    elif "Color" in result_df:
        result_df["Detected_Color"] = result_df.apply(lambda row: detect_color(row["Color"]), axis=1)
        result_df["displayName"] = result_df.apply(lambda row: generate_display_name(row["Color"], row['Detected_Color'], row["Detected_Origin"], row['Indication'], row['oldHeat']), axis=1)

    result_df["Detected_Cut"] = result_df["Cut"].apply(detect_cut)
    result_df["Detected_Shape"] = result_df["Shape"].apply(detect_shape)
    result_df["Reformatted_issuedDate"] = result_df["issuedDate"].apply(reformat_issued_date)
    result_df["Mogok"] = result_df["Origin"].apply(detect_mogok)
    result_df["Indication"] = result_df["indications"].apply(generate_indication)
    result_df = extract_cert_info(result_df, 'certNo')
    result_df["carat"] = result_df["Carat"].apply(convert_carat_to_numeric)
    result_df[["length", "width", "height"]] = result_df["Dimensions"].apply(convert_dimension).apply(pd.Series)
    result_df = rename_identification_to_stone(result_df)

    result_df = result_df[[
    "certName",
    "certNO",
    "displayName",
    "Stone",
    "Detected_Color",
    "Detected_Origin",
    "Reformatted_issuedDate",
    "Indication",
    "oldHeat",
    "Mogok",
    "Detected_Cut",
    "Detected_Shape",
    "carat",
    "length",
    "width",
    "height"
    ]]
    
    return result_df

# Specify the folder containing the images
# folder_path = r'C:\Users\kan43\Downloads\Cert Scraping Test'

# Specify the file pattern you want to filter
file_pattern = "-01_SSEF"

# Create a Streamlit file uploader for the zip file
zip_file = st.file_uploader("Upload a ZIP file containing images", type=["zip"])

if zip_file is not None:
    # Extract the uploaded ZIP file
    with zipfile.ZipFile(zip_file) as zip_data:
        df_list = []

        for image_file in zip_data.namelist():
            if file_pattern in image_file:
                filename_without_suffix = image_file.split('-')[0]
                try:
                    # Read the image
                    with zip_data.open(image_file) as file:
                        img_data = io.BytesIO(file.read())
                        img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 0)
                        
                        # Process the image and perform data processing
                        df = extract_gemstone_info(img)
                        result_df = perform_data_processing(df)
    
                        result_df['StoneID'] = filename_without_suffix
                        result_df["StoneID"] = result_df["StoneID"].str.split("/")
                        # Get the last part of each split
                        result_df["StoneID"] = result_df["StoneID"].str.get(-1)
    
                        result_df = result_df[[
                            "certName",
                            "certNO",
                            "StoneID",
                            "displayName",
                            "Stone",
                            "Detected_Color",
                            "Detected_Origin",
                            "Reformatted_issuedDate",
                            "Indication",
                            "oldHeat",
                            "Mogok",
                            "Detected_Cut",
                            "Detected_Shape",
                            "carat",
                            "length",
                            "width",
                            "height",
                        ]]
                        result_df = result_df.rename(columns={
                            "Detected_Color": "Color",
                            "Detected_Origin": "Origin",
                            "Reformatted_issuedDate": "issuedDate",
                            "Detected_Cut": "Cut",
                            "Detected_Shape": "Shape"
                        })
    
                        # Append the DataFrame to the list
                        df_list.append(result_df)
                except Exception as e:
                    # Handle errors for this image, you can log or print the error message
                    st.error(f"Error processing image {image_file}: {str(e)}")
                    pass  # Skip to the next image

        # Concatenate all DataFrames into one large DataFrame
        final_df = pd.concat(df_list, ignore_index=True)

        # Display the final DataFrame
        st.write(final_df)


        csv_data = final_df.to_csv(index=False, float_format="%.2f").encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="Cert.csv",
            key="download-button"
        )
