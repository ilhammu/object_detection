import streamlit as st
from PIL import Image
from enhance_image import Enhance
# from pandas.compat import StringIO
import io
def main():
    enc = Enhance()

    st.title('Image Enhancement')
    
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write('')
        st.write('Processing...')

        # Send the uploaded file to Flask API for processing
        response_data = enc.process_image(uploaded_file)

        # Display the original and enhanced images
        input_image = Image.open(uploaded_file)
        enhanced_image = Image.open(io.BytesIO(response_data['enhanced_image_bytes']))

        st.image([input_image, enhanced_image], caption=['Original Image', 'Enhanced Image'], use_column_width=True)
        st.write('Finish')
if __name__ == '__main__':
    main()
