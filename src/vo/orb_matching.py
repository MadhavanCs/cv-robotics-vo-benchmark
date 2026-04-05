import cv2
import numpy as np

def main():
    img1 = cv2.imread("data/images/test.jpeg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("data/images/test2.jpeg", cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("One or both images not found")
        return

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Filter good matches using distance threshold
    good_matches = [m for m in matches if m.distance < 50]

    print(f"Total matches: {len(matches)}")
    print(f"Good matches: {len(good_matches)}")

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None,
                                   matchColor=(0, 255, 0),    # green = good
                                   singlePointColor=(255, 0, 0))  # blue = unmatched

    img_matches = cv2.resize(img_matches, (1280, 480))
    cv2.imshow("ORB Matches", img_matches)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()