import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_pose(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 50]

    if len(good_matches) < 8:
        print("Not enough matches")
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    h, w = img1.shape
    focal = w
    cx, cy = w / 2, h / 2
    K = np.array([[focal, 0, cx],
                  [0, focal, cy],
                  [0,     0,  1]], dtype=np.float64)

    E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    return R, t

def main():
    # Load your images
    images = []
    image_files = [
        "data/images/test.jpeg",
        "data/images/test2.jpeg",
    ]

    for f in image_files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load {f}")
            return
        images.append(img)

    # Track camera position
    trajectory = [[0, 0, 0]]  # start at origin
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))

    for i in range(len(images) - 1):
        R, t = estimate_pose(images[i], images[i+1])
        if R is None:
            continue

        # Accumulate pose
        t_total = t_total + R_total @ t
        R_total = R @ R_total

        x = t_total[0, 0]
        z = t_total[2, 0]
        trajectory.append([x, 0, z])
        print(f"Frame {i+1}: x={x:.3f}, z={z:.3f}")

    # Plot trajectory
    trajectory = np.array(trajectory)
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 2], 'bo-', markersize=8)
    plt.scatter(trajectory[0, 0], trajectory[0, 2], c='green', s=100, label='Start', zorder=5)
    plt.scatter(trajectory[-1, 0], trajectory[-1, 2], c='red', s=100, label='End', zorder=5)
    plt.title("Camera Trajectory")
    plt.xlabel("X (sideways)")
    plt.ylabel("Z (forward)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/trajectory.png")
    plt.show()
    print("\nTrajectory saved to results/trajectory.png")

if __name__ == "__main__":
    main()