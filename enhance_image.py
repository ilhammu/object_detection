# import json
from PIL import Image, ImageFilter
import numpy as np
import io

class Enhance:
    def multiscale_retinex(self, image, sigma_list=[100, 200, 300]):
        retinex_image = np.zeros_like(image, dtype=np.float32)

        for sigma in sigma_list:
            log_image = np.log1p(image.astype(np.float32))
            gaussian = Image.fromarray((log_image * 255).astype('uint8')).filter(
                ImageFilter.GaussianBlur(radius=sigma)
            )
            retinex_image += np.array(log_image * 255) - np.array(gaussian)

        retinex_image /= len(sigma_list)
        retinex_image = np.exp(retinex_image / 255.0) - 1.0

        # Normalization of the image
        retinex_image_normalized = (retinex_image - np.min(retinex_image)) / (np.max(retinex_image) - np.min(retinex_image)) * 255
        retinex_image_normalized = retinex_image_normalized.astype('uint8')
        return retinex_image_normalized

    def process_image(self, uploaded_file):
        # Read the image from the UploadedFile
        input_image = Image.open(uploaded_file)
        input_image_np = np.array(input_image)

        # Perform image processing
        enhanced_image_np = self.multiscale_retinex(input_image_np)

        # Convert NumPy array back to PIL Image
        enhanced_image = Image.fromarray(enhanced_image_np)

        # Save the enhanced image to bytes
        enhanced_image_bytes = io.BytesIO()
        enhanced_image.save(enhanced_image_bytes, format='JPEG')
        enhanced_image_bytes.seek(0)

        # Prepare response data
        response_data = {
            'input_image_path': uploaded_file.name,
            'enhanced_image_bytes': enhanced_image_bytes.getvalue(),
        }

        return response_data
