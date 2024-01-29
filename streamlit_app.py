# streamlit_app.py

import subprocess
import streamlit as st
from PIL import Image
from enhance_image import Enhance

def install_requirements():

    # Run the pip install command
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

def main():

    enc = Enhance();
    process_image = enc.process_image()

    install_requirements()
    st.title('Image Enhancement with Multiscale Retinex')
    
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write('')
        st.write('Processing...')

        # Save the uploaded file
        with open('input_image.jpg', 'wb') as f:
            f.write(uploaded_file.read())

        # Send the image to Flask API for processing
        response_data = process_image('input_image.jpg')

        # Display the original and enhanced images
        input_image = Image.open(response_data['input_image_path'])
        enhanced_image = Image.open(response_data['enhanced_image_path'])

        st.image([input_image, enhanced_image], caption=['Original Image', 'Enhanced Image'], use_column_width=True)

if __name__ == '__main__':
    main()
