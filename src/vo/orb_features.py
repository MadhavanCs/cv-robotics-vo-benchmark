import cv2

def main():
    img = cv2.imread("data/images/test.jpg", cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not found")
        return

    orb = cv2.ORB_create()
    keypoints = orb.detect(img, None)
    img_kp = cv2.drawKeypoints(img, keypoints, None)

    cv2.imshow("ORB Keypoints", img_kp)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
