import cv2
import os

def preprocess_fingerprint(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def compute_match_score(image1, image2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    total_distance = sum([m.distance for m in matches])
    avg_distance = total_distance / len(matches) if matches else float('inf')
    match_score = 1 - (avg_distance / 100) 
    
    return match_score, kp1, kp2, matches

def match_fingerprint_to_database(query_image_path, database_folder, threshold=0.5):
    query_image = cv2.imread(query_image_path)
    query_image = preprocess_fingerprint(query_image)

    for filename in os.listdir(database_folder):
        db_image_path = os.path.join(database_folder, filename)
        db_image = cv2.imread(db_image_path)
        
        if db_image is None:
            print(f"Failed to load image: {db_image_path}")
            continue
        
        db_image = preprocess_fingerprint(db_image)
        match_score, kp1, kp2, matches = compute_match_score(query_image, db_image)
        print(f"Comparing with {filename}: Match Score = {match_score * 100:.2f}%")

        match_img = cv2.drawMatches(query_image, kp1, db_image, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow(f'Match with {filename}', match_img)
        cv2.waitKey(0) 

        if match_score >= threshold:
            print(f"Match found with {filename}!")
            cv2.destroyAllWindows()
            return filename 

    print("No match found.")
    cv2.destroyAllWindows()
    return None

query_image_path = '/path/to/query/image.jpg'
database_folder = '/path/to/database/folder'

match_fingerprint_to_database(query_image_path, database_folder)