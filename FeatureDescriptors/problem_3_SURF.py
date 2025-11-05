
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def detect_and_draw_keypoints(img_color, img_gray, method='SURF', hessian_threshold=400, sift_threshold=0.04):
    if method == 'SURF':
        # -- SURF Blob Detection
        detector = cv.xfeatures2d_SURF.create(hessianThreshold=hessian_threshold)
    elif method == 'SIFT':
        # -- SIFT Blob Detection
        detector = cv.SIFT_create(contrastThreshold=sift_threshold)
    else:
        raise ValueError("Unsupported method. Use 'SURF' or 'SIFT'.")

    keypoints = detector.detect(img_gray, None)

    # -- Draw keypoints
    img_keypoints = cv.drawKeypoints(img_color, keypoints, None, (255, 0, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img_keypoints, len(keypoints)

def main():
    image_filename = 'D:\\UB\\CPEG-585 COMPUTER VISION\\Assignment_4\\sunflower_small.jpg'
    img_color = cv.imread(image_filename)
    img_color = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    
    
    #-- Step 1: Detect the keypoints using SURF Detector
    hessian_threshold   = 400        #Change the value of the minimum Hessian value to detect only the blob of the sunflower core
    # detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    # keypoints = detector.detect(img)
    img_surf_keypoints, surf_count = detect_and_draw_keypoints(img_color, img_gray, method='SURF', hessian_threshold=hessian_threshold)

    sift_threshold = 0.04  # Adjust for sensitivity
    img_sift_keypoints, sift_count = detect_and_draw_keypoints(img_color, img_gray, method='SIFT', sift_threshold=sift_threshold)
    
    #-- Draw keypoints
    #img_keypoints = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #img_keypoints = cv.drawKeypoints(img_color,keypoints,None,(255,0,0),4)
    # Add your code here to save the output marked image
    
    print(f"SURF detected {surf_count} keypoints")
    print(f"SIFT detected {sift_count} keypoints")

    # Plot SURF
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("SURF Keypoints")
    plt.imshow(img_surf_keypoints)
    plt.axis('off')

    # Plot SIFT
    plt.subplot(1, 2, 2)
    plt.title("SIFT Keypoints")
    plt.imshow(img_sift_keypoints)
    plt.axis('off')
    plt.show()

    # Save images
    cv.imwrite('D:\\UB\\CPEG-585 COMPUTER VISION\\Assignment_4\\sunflower_surf_keypoints.jpg', cv.cvtColor(img_surf_keypoints, cv.COLOR_RGB2BGR))
    cv.imwrite('D:\\UB\\CPEG-585 COMPUTER VISION\\Assignment_4\\sunflower_sift_keypoints.jpg', cv.cvtColor(img_sift_keypoints, cv.COLOR_RGB2BGR))


    #-- Show detected (drawn) keypoints
    #plt.imshow(img_keypoints)
    # plt.axis('off')
    # plt.show()

if __name__ == '__main__':
    main()
    pass
