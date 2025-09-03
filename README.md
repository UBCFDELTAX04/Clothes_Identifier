Clothes Identifier: A Real-time Fashion-MNIST Classifier Project Description This is an innovative, real-time application that uses a deep learning model to classify clothing items. Built with Python and the Streamlit framework, this project allows users to either upload an image or use their live camera feed to identify different articles of clothing (like t-shirts, dresses, or bags). The app is a great example of an Internet of Things (IoT) application, where data from a local device (a camera) is processed by a powerful cloud-based machine learning model.

A key feature of this application is its robust preprocessing pipeline. It includes an interactive image cropper and an automatic background removal function, which significantly improves the accuracy of predictions by ensuring the model only sees the clothing item itself.

Features Two Input Modes: Upload an image file or use a live camera feed.

Interactive Cropper: Visually crop and zoom into the item of clothing for precise classification.

Automatic Background Removal: Uses the rembg library to isolate the clothing item from its background.

Fashion-MNIST Classification: Predicts the category of the clothing item (e.g., T-shirt, Trouser, Sneaker) using a pre-trained Convolutional Neural Network (CNN) model.

Installation and Setup To get this app running on your local machine, follow these simple steps.

Clone the repository:

git clone https://github.com/UBCFDELTAX04/Clothes_Identifier.git cd Clothes_Identifier

Create a virtual environment (recommended):

python -m venv venv

Activate the virtual environment:

On Windows: venv\Scripts\activate

On macOS and Linux: source venv/bin/activate

Install dependencies: The project's dependencies are listed in requirements.txt. Install them using pip:

pip install -r requirements.txt

Download the model: This app requires a pre-trained model file named fashion_mnist_cnn_model.keras. You'll need to download this file and place it in the same directory as app.py.

Note: You can train this model yourself or find a pre-trained model online.

How to Run the App Once you have the dependencies and the model file in place, you can run the app with this single command:

streamlit run app.py

Your browser will automatically open a new tab showing the running application.

Deployment on Streamlit Cloud If you're deploying this app on Streamlit Cloud, remember to include the following two files in the root of your repository to ensure a successful deployment:

requirements.txt: Lists all Python dependencies.

runtime.txt: Specifies the Python version for the environment. Add python-3.11 to this file to ensure compatibility.

Contributing This project is open for contributions! If you have suggestions for improvements, new features, or bug fixes, feel free to open an issue or submit a pull request.

Made with ❤️ by UBCFDELTAX04
