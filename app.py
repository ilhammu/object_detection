# Import required libraries
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageOps
import io
import base64

# Load the object detection model from TensorFlow Hub
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

# Helper function to run the detector
def run_detector(detector, img):
    img_array = np.array(img)
    converted_img = tf.image.convert_image_dtype(img_array, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)

    result = {key: value.numpy() for key, value in result.items()}

    st.write("Found %d objects." % len(result["detection_scores"]))

    image_with_boxes = draw_boxes(
        img_array, result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"])

    return image_with_boxes

# Helper function to draw bounding boxes
def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image

# Streamlit app
def main():
    st.title("Object Detection with TensorFlow Hub and Streamlit")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        img = Image.open(uploaded_file)
        img = ImageOps.fit(img, (600, 600), Image.ANTIALIAS)

        result_image = run_detector(detector, img)

        st.image(result_image, caption="Result", use_column_width=True)

        # Convert result image to base64 for download link
        buffered = io.BytesIO()
        result_image_pil = Image.fromarray(result_image)
        result_image_pil.save(buffered, format="JPEG")
        result_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Provide download link for the result image
        st.markdown(get_download_link(result_image_base64), unsafe_allow_html=True)

# Helper function to create a download link for the result image
def get_download_link(image_base64):
    href = f'<a href="data:file/jpg;base64,{image_base64}" download="result_image.jpg">Download Result Image</a>'
    return href

if __name__ == '__main__':
    main()
