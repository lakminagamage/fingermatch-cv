import cv2

def extract_orb_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    return keypoints, descriptors

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    return keypoints, descriptors

if __name__ == "__main__":
    image = cv2.imread('path/to/image.png', cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = extract_orb_features(image)
    
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    
    cv2.imshow('ORB Keypoints', image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
