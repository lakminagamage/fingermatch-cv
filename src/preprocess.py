import cv2
import numpy as np

def preprocess_fingerprint(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return opened

if __name__ == "__main__":
    image = cv2.imread('../path/to/image.png')
    processed_image = preprocess_fingerprint(image)
    cv2.imshow('Processed Fingerprint', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
