import streamlit as st
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import uuid

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


# Define the layout of the app
st.set_page_config(page_title="Clock Drawing Test", page_icon=":medical_symbol:")

st.title("Clock Drawing Test")

st.header(
    "AI-based Clock Drawing Test for Dementia - 85% Accurate Detection Algorithm."
)

st.write(
    "Try drawing a Clock and watch how an AI Model will detect Dementia."
)

st.caption(
    "The application will infer the one label out of 2 labels, as follows: Healthy, Dementia."
)

st.warning(
    "Warning: Do not click Submit Sketch button before drawing clock on below Canvas."
)

with st.sidebar:
    img = Image.open("./Images/dementia.png")
    st.image(img)
    st.subheader("About Dementia")
    link_text = "How the Clock-Drawing Test Screens for Dementia"
    link_url = "https://www.verywellhealth.com/the-clock-drawing-test-98619"
    st.write(
        "Dementia is the loss of cognitive functioning — thinking, remembering, and reasoning — to such an extent that it interferes with a person's daily life and activities. Some people with dementia cannot control their emotions, and their personalities may change."
    )
    st.markdown(f"[{link_text}]({link_url})")
    st.header("Dataset")
    img = Image.open("./Images/clock_drawing_test_classification.png")
    st.image(img)
    st.header("Drawing Canvas Configurations")

# Specify canvas parameters in application
drawing_mode = "freedraw"

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Split the layout into two columns
col1, col2 = st.columns(2)

# Define the canvas size
canvas_size = 345

with col1:
    # Create a canvas component
    st.subheader("Drawable Canvas")
    canvas_image = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        width=canvas_size,
        height=canvas_size,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        key="canvas",
    )

with col2:
    st.subheader("Preview")
    if canvas_image.image_data is not None:
        # Get the numpy array (4-channel RGBA 100,100,4)
        input_numpy_array = np.array(canvas_image.image_data)
        # Get the RGBA PIL image
        input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
        st.image(input_image, use_column_width=True)


def generate_user_input_filename():
    unique_id = uuid.uuid4().hex
    filename = f"user_input_{unique_id}.png"
    return filename


def predict_dementia(img_path):
    best_model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Get the numpy array (4-channel RGBA 100,100,4)
    input_numpy_array = np.array(img_path.image_data)

    # Get the RGBA PIL image
    input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")

    # Generate a unique filename for the user input
    user_input_filename = generate_user_input_filename()

    # Save the image with the generated filename
    input_image.save(user_input_filename)
    print("Image Saved!")   

    # Replace this with the path to your image
    image = Image.open(user_input_filename).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = best_model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    Detection_Result = f"The model has detected {class_name[2:]}, with Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%."
    os.remove(user_input_filename)
    print("Image Removed!")
    return Detection_Result, prediction


submit = st.button(label="Submit Sketch")
if submit:
    st.subheader("Output")
    classified_label, prediction = predict_dementia(canvas_image)
    with st.spinner(text="This may take a moment..."):
        st.write(classified_label)

        class_names = open("labels.txt", "r").readlines()

        data = {
            "Class": class_names,
            "Confidence Score": prediction[0],
        }

        df = pd.DataFrame(data)

        df["Confidence Score"] = df["Confidence Score"].apply(
            lambda x: f"{str(np.round(x*100))[:-2]}%"
        )

        df["Class"] = df["Class"].apply(lambda x: x.split(" ", 1)[1])

        st.subheader("Confidence Scores on other classes:")
        st.write(df)

footer = """
<div style="text-align: center; font-size: medium; margin-top:50px;">
    If you find Dementia useful or interesting, please consider starring it on GitHub.
    <hr>
    <a href="https://github.com/sakuriee/AI-based-Clock-Drawing-Test-for-Dementia" target="_blank">
    <img src="https://img.shields.io/github/stars/sakuriee/AI-based-Clock-Drawing-Test-for-Dementia.svg?style=social" alt="GitHub stars">
  </a>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
