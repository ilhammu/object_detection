class Enhance:
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from flask import jsonify

    def multiscale_retinex(self, image, sigma_list=[100, 200, 300]):
        retinex_image = self.self.np.zeros_like(image, dtype=self.self.np.float32)

        for sigma in sigma_list:
            log_image = self.np.log1p(image.astype(self.np.float32))
            gaussian = self.cv2.GaussianBlur(log_image, (0, 0), sigma)
            retinex_image += log_image - gaussian

        retinex_image /= len(sigma_list)
        retinex_image = self.np.exp(retinex_image) - 1.0

        # Normalisasi gambar
        retinex_image_normalized = (retinex_image - self.np.min(retinex_image)) / (self.np.max(retinex_image) - self.np.min(retinex_image)) * 255
        #retinex_image_normalized = retinex_image_normalized.astype(self.np.uint8)
        retinex_image_normalized = retinex_image
        return retinex_image
    
    def process_image(self,file):

        # Save the received file
        file.save('input_image.jpg')

        # Read the image and perform image processing
        input_image = self.cv2.imread('input_image.jpg')
        enhanced_image = self.multiscale_retinex(input_image)

        # Save the enhanced image
        output_filename = 'enhanced_image.jpg'
        self.cv2.imwrite(output_filename, enhanced_image)

        # Prepare response data
        response_data = {
            'input_image_path': 'input_image.jpg',
            'enhanced_image_path': output_filename,
        }

        return self.jsonify(response_data)