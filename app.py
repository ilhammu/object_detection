from flask import Flask, request, jsonify
import cv2
import numpy as np
from matplotlib import pyplot as plt
from enhance_image import Enhance
app = Flask(__name__)

# # Fungsi untuk mengonversi dan menampilkan gambar
# def process_and_display_image(image_path):
#     # Baca gambar
#     input_image = cv2.imread(image_path)

#     # Konversi gambar ke skala keabuan (grayscale)
#     gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

#     # Ditingkatkan dengan MSR
#     enhanced_image = multiscale_retinex(input_image)

#     # Tampilkan gambar asli
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
#     plt.title('Gambar Asli')

#     # Tampilkan gambar skala keabuan
#     plt.subplot(1, 3, 2)
#     plt.imshow(gray_image, cmap='gray')
#     plt.title('Gambar Skala Keabuan')

#     # Tampilkan gambar yang telah ditingkatkan dengan MSR
#     plt.subplot(1, 3, 3)
#     plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
#     output_filename = 'enhanced_image.jpg'
#     cv2.imwrite(output_filename, enhanced_image)
#     plt.title('Hasil Enhancement MSR')
#     plt.show()

# # Fungsi Multiscale Retinex (MSR)
# def multiscale_retinex(image, sigma_list=[100, 200, 300]):
    # retinex_image = np.zeros_like(image, dtype=np.float32)

    # for sigma in sigma_list:
    #     log_image = np.log1p(image.astype(np.float32))
    #     gaussian = cv2.GaussianBlur(log_image, (0, 0), sigma)
    #     retinex_image += log_image - gaussian

    # retinex_image /= len(sigma_list)
    # retinex_image = np.exp(retinex_image) - 1.0

    # # Normalisasi gambar
    # retinex_image_normalized = (retinex_image - np.min(retinex_image)) / (np.max(retinex_image) - np.min(retinex_image)) * 255
    # #retinex_image_normalized = retinex_image_normalized.astype(np.uint8)
    # retinex_image_normalized = retinex_image
    # return retinex_image


@app.route('/process_image', methods=['POST'])
def process_image():
    enc = Enhance();
    multiscale_retinex = enc.multiscale_retinex()
    # Check if the POST request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the received file
    file.save('input_image.jpg')

    # Read the image and perform image processing
    input_image = cv2.imread('input_image.jpg')
    enhanced_image = multiscale_retinex(input_image)

    # Save the enhanced image
    output_filename = 'enhanced_image.jpg'
    cv2.imwrite(output_filename, enhanced_image)

    # Prepare response data
    response_data = {
        'input_image_path': 'input_image.jpg',
        'enhanced_image_path': output_filename,
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(port=5000)