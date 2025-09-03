import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
# To run this code, you'll need to install the following libraries:
# pip install streamlit tensorflow pillow rembg onnxruntime

# --- IMPORTANT: Ensure you have the rembg library and its dependencies installed. ---
# --- You also need the 'fashion_mnist_cnn_model.keras' file in the same directory. ---
from rembg import remove

# Define the class names for the Fashion-MNIST dataset
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model():
    """Loads the pre-trained Keras model from disk."""
    model_path = 'fashion_mnist_cnn_model.keras'
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: The model file '{model_path}' was not found.")
        st.warning("Please make sure you have the 'fashion_mnist_cnn_model.keras' file in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def preprocess_image(image):
    """
    Preprocesses the user-uploaded or captured image for the model.
    This includes background removal, converting to grayscale, resizing,
    inverting colors, and normalization.

    Args:
        image: PIL Image object.

    Returns:
        A NumPy array of the processed image, ready for prediction.
    """
    # 1. Remove the background from the image
    # We convert the image to RGB first for best results with rembg
    try:
        image_no_bg = remove(image.convert("RGB"))
    except Exception as e:
        st.warning(f"Could not remove background. The original image will be used. Error: {e}")
        image_no_bg = image

    # 2. Convert to grayscale and resize to 28x28
    img = image_no_bg.convert('L')
    img = img.resize((28, 28))

    # 3. Convert to a NumPy array
    img_array = np.array(img)

    # 4. Invert the colors (Fashion-MNIST is dark on a light background)
    img_array = 255 - img_array

    # 5. Normalize pixel values to be between 0 and 1
    img_array = img_array.astype('float32') / 255.0

    # 6. Add channel and batch dimensions for the model
    return np.expand_dims(np.expand_dims(img_array, axis=0), axis=-1)

# --- Main Streamlit App ---

st.set_page_config(page_title="Fashion-MNIST Classifier", page_icon="👚", layout="centered")

st.title("Fashion-MNIST Image Classifier")
st.markdown("Choose an input method and I'll predict the article of clothing!")
st.markdown("---")

# Use a sidebar to let the user choose the input method
st.sidebar.title("Input Method")
input_method = st.sidebar.radio("Select an option:", ("Upload an image", "Capture from camera"))

image = None

if input_method == "Upload an image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
elif input_method == "Capture from camera":
    captured_file = st.camera_input("Take a picture")
    if captured_file:
        image = Image.open(io.BytesIO(captured_file.getvalue()))

if image is not None:
    # Load the pre-trained model
    model = load_model()

    if model:
        # Display a spinner while processing
        with st.spinner('Analyzing image...'):
            try:
                st.image(image, caption='Original Image', use_column_width=True)

                # Preprocess the image for the model, including background removal
                processed_image = preprocess_image(image)

                # Make a prediction
                prediction = model.predict(processed_image, verbose=0)
                predicted_class_index = np.argmax(prediction)
                predicted_label = CLASS_NAMES[predicted_class_index]
                confidence = np.max(prediction)

                # Display the processed image to show the user what the model sees
                st.subheader("Processed Image")
                # Need to convert back to PIL Image for display
                display_img = Image.fromarray(np.uint8((255 - processed_image[0, :, :, 0] * 255)))
                st.image(display_img, caption="Background removed and resized for the model.", use_column_width=False)
                st.markdown("---")

                # Display the prediction using a success box for visibility
                st.success(f"Prediction: **{predicted_label}**")
                st.info(f"Confidence: **{confidence:.2f}**")

                # Show the prediction probabilities
                st.subheader("Prediction Probabilities")

                import pandas as pd
                df_probabilities = pd.DataFrame({
                    "Category": CLASS_NAMES,
                    "Probability": prediction[0]
                })

                df_probabilities = df_probabilities.sort_values(by="Probability", ascending=False)
                st.dataframe(df_probabilities, hide_index=True, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
