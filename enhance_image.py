from flask import jsonify
import cv2
import numpy as np

class Enhance:
    def multiscale_retinex(self, image, sigma_list=[100, 200, 300]):
        retinex_image = np.zeros_like(image, dtype=np.float32)

        for sigma in sigma_list:
            log_image = np.log1p(image.astype(np.float32))
            gaussian = cv2.GaussianBlur(log_image, (0, 0), sigma)
            retinex_image += log_image - gaussian

        retinex_image /= len(sigma_list)
        retinex_image = np.exp(retinex_image) - 1.0

        # Normalization of the image
        retinex_image_normalized = (retinex_image - np.min(retinex_image)) / (np.max(retinex_image) - np.min(retinex_image)) * 255
        retinex_image_normalized = retinex_image
        return retinex_image_normalized

    def process_image(self, file):
        # Save the received file
        file.save('input_image.jpg')

        # Read the image and perform image processing
        input_image = cv2.imread('input_image.jpg')
        enhanced_image = self.multiscale_retinex(input_image)

        # Save the enhanced image
        output_filename = 'enhanced_image.jpg'
        cv2.imwrite(output_filename, enhanced_image)

        # Prepare response data
        response_data = {
            'input_image_path': 'input_image.jpg',
            'enhanced_image_path': output_filename,
        }

        return jsonify(response_data)
