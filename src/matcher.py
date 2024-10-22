import cv2

def match_fingerprints(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches

if __name__ == "__main__":
    image1 = cv2.imread('../images/fingerprint1.png', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('../images/fingerprint2.png', cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    _, descriptors1 = orb.detectAndCompute(image1, None)
    _, descriptors2 = orb.detectAndCompute(image2, None)

    matches = match_fingerprints(descriptors1, descriptors2)
    matched_image = cv2.drawMatches(image1, None, image2, None, matches[:10], None, flags=2)
    
    cv2.imshow('Matches', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
