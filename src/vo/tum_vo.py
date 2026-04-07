import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_tum_images(dataset_path, max_frames=20):
    rgb_txt = os.path.join(dataset_path, "rgb.txt")
    images = []
    
    with open(rgb_txt, 'r') as f:
        lines = f.readlines()
    
    
    lines = [l.strip() for l in lines if not l.startswith('#')]
    
    for line in lines[:max_frames]:
        parts = line.split()
        if len(parts) < 2:
            continue
        img_path = os.path.join(dataset_path, parts[1])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    
    print(f"Loaded {len(images)} frames")
    return images

def estimate_pose(img1, img2):
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 50]

    if len(good_matches) < 8:
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    h, w = img1.shape
    focal = 525.0  # TUM fr1 camera focal length
    cx, cy = 319.5, 239.5  # TUM fr1 principal point
    K = np.array([[focal, 0, cx],
                  [0, focal, cy],
                  [0,     0,  1]], dtype=np.float64)

    E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None
        
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def main():
    dataset_path = "data/tum/fr1_desk"
    
    print("Loading TUM frames...")
    images = load_tum_images(dataset_path, max_frames=20)
    
    if len(images) < 2:
        print("Not enough images loaded")
        return

    # Track trajectory
    trajectory = [[0, 0, 0]]
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))

    for i in range(len(images) - 1):
        R, t = estimate_pose(images[i], images[i+1])
        if R is None:
            print(f"Skipping frame {i}")
            continue

        t_total = t_total + R_total @ t
        R_total = R @ R_total

        x = t_total[0, 0]
        z = t_total[2, 0]
        trajectory.append([x, 0, z])
        print(f"Frame {i+1}/{len(images)-1}: x={x:.3f}, z={z:.3f}")

    # Plot
    trajectory = np.array(trajectory)
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 2], 'b-o', markersize=5)
    plt.scatter(trajectory[0, 0], trajectory[0, 2], c='green', s=150, label='Start', zorder=5)
    plt.scatter(trajectory[-1, 0], trajectory[-1, 2], c='red', s=150, label='End', zorder=5)
    plt.title("TUM Dataset - Camera Trajectory (fr1/desk)")
    plt.xlabel("X (sideways)")
    plt.ylabel("Z (forward)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/tum_trajectory.png")
    plt.show()
    print("\nSaved to results/tum_trajectory.png")

if __name__ == "__main__":
    main()