import cv2
import os


def preprocess(img_path):
    output_dir = r"C:\\Users\\tusha\Dropbox\\My PC (LAPTOP-TQAVKKGE)\Desktop\\Main"

    image_size = (128, 32)
    threshold_value = 127
    denoising_strength = 10

    image_path = img_path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, image_size)

    _, binary_image = cv2.threshold(
        image, threshold_value, 1, cv2.THRESH_BINARY)

    equalized_image = cv2.equalizeHist(binary_image)

    denoised_image = cv2.fastNlMeansDenoising(
        equalized_image, None, denoising_strength, 7, 21)

    output_path = os.path.join(output_dir, "test.png")
    cv2.imwrite(output_path, denoised_image)

    print(f'Preprocessed image saved: {output_path}')
