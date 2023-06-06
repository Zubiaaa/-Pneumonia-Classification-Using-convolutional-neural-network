# Deploying-Machine-Learning-App-on-Heroku

Develop an image classification model without writing a single line of code:
1.	Teachable Machine
2.	Kaggle X-ray dataset
Let's take https://teachablemachine.withgoogle.com/ to create our neural network model. Teachable Machine is a web-based tool that makes creating machine learning models fast, easy, and accessible to everyone. It is an AutoML tool from Google.
The X-ray dataset is taken from Kaggle.
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
X-Ray Image Classification App: The model is trained with the above dataset. When you upload an X-ray image to the app then it classifies whether the X-ray is normal or it has pneumonia symptoms.
Steps to create the model and app:
1.	Download the dataset from Kaggle. The link is https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
The size of the dataset is 1.15 GB and contains around 5856 images .
2.	Go to teachablemachine and the URL is:
https://teachablemachine.withgoogle.com/train/image
3.	There is no need for any preprocessing of the data. Upload all the normal images to Class1. Upload all the pneumonia images to Class 2. I have considered only the training image folder.

4.	After uploading all the images then choose Train Model. I just changed the Batch Size from 16 to 128 in the Advanced Settings. I maintained the same default 50 Epochs and default learning rate.
5.	After the model is Trained then export the model (optionally: also, you can test the model). I only uploaded the training folder images and trained the model.
6.	Export the model - Choose the Tensorflow and choose Keras. Then press Download my model.

Develop a machine learning app using Stremlit:
With minimal code, you can create an app using Streamlit. check out
https://streamlit.io/
1.	Install Streamlit.
2.	The code-snippet to test the Keras model is provided in the teachbalemachine and it is:
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

#Disable scientific notation for clarity
np.set_printoptions(suppress=True)

#Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

#Create the array of the right shape to feed into the keras model
#The 'length' or number of images you can put into the array is
#determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#Replace this with the path to your image
image = Image.open('test_photo.jpg')

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)

#display the resized image
image.show()

#Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

#Load the image into the array
data[0] = normalized_image_array

#run the inference
prediction = model.predict(data)
print(prediction)

3.	Now you use the Streamlit file upload and also add some headings:

st.title("Image Classification with Teachable Machine Learning")
st.header("Normal X Ray Vs Pneumonia X Ray")
st.text("Upload a X Ray to detect it is normal or has pneumonia")
#file upload and handling logic
uploaded_file = st.file_uploader("Choose a X Ray Image", type="jpeg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
#image = Image.open(img_name).convert('RGB')
    st.image(image, caption='Uploaded a X Ray IMage.', use_column_width=True)
    st.write("")
    st.write("Classifying a X Ray Image - Normal Vs Pneumonia.........hold tight")
    label = teachable_machine_classification(image, 'C:\BPC_DOCS\IUB\Projects_Medium\converted_keras\keras_model.h5')
    if label == 1:
        st.write("This X ray looks like having pneumonia.It has abnormal opacification.Needs further investigation by a Radiologist/Doctor.")
    else:
        st.write("Hooray!! This X ray looks normal.This X ray depicts clear lungs without any areas of abnormal opacification in the image")

4.	To execute the app call the file — Refer the below code. The file name is xray.py:
streamlit run xray.py

5.	Test the app locally with an x-ray image from the normal category and another one from the pneumonia category.
6.	Create a GitHub repository and upload all the 7 files (.slugignore, keras_model.h5, labels.txt, Procfile, requirements.txt, setup.sh, and xray.py). labels.txt file and keras_model.h5 are optional files (not required to upload them on GitHub to deploy the app on Heroku).
7.	The last step is to deploy the app on Heroku. Go to the following link:
https://dashboard.heroku.com/apps
8.	Create an app in Heroku. With the app, go to the Deploy tab. Go to the GitHub sub-tab. Connect your GitHub repository (where you uploaded all the files).
9.	Click on Deploy Branch to deploy the app on Heroku. 
10.	After successfully deploying, click on view to open the app.
