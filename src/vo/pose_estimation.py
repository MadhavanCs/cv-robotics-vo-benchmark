import cv2
import numpy as np

def main():
    img1 = cv2.imread("data/images/test.jpeg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("data/images/test2.jpeg", cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("One or both images not found")
        return

    # Detect and match features
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 50]

    print(f"Good matches: {len(good_matches)}")

    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Camera intrinsics (approximate for unknown camera)
    h, w = img1.shape
    focal = w  # rough estimate
    cx, cy = w / 2, h / 2
    K = np.array([[focal, 0, cx],
                  [0, focal, cy],
                  [0,     0,  1]], dtype=np.float64)

    # Estimate Essential Matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover camera pose
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    print("\nRotation Matrix R:")
    print(np.round(R, 3))

    print("\nTranslation Vector t:")
    print(np.round(t, 3))

if __name__ == "__main__":
    main()