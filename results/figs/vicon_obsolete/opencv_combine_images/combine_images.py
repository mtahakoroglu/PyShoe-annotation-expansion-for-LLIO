import cv2
import os
import numpy as np
# from PIL import Image

expNumber = 43
trajectoryText = f"trajectory_exp_{expNumber}.png"
trajectoryCorrectedText = f"trajectory_exp_{expNumber}_corrected.png"
zvlabelsText = f"zv_labels_exp_{expNumber}.png"
zvlabelsCorrectedText = f"zv_labels_exp_{expNumber}_corrected.png"
img1 = cv2.imread(trajectoryText)
img2 = cv2.imread(zvlabelsText)
img3 = cv2.imread(trajectoryCorrectedText)
img4 = cv2.imread(zvlabelsCorrectedText)

if img1.shape[0] > img2.shape[0]:
    s = img2.shape[0] / img1.shape[0]
    img1 = cv2.resize(img1, (int(s*img1.shape[1]), int(s*img1.shape[0])), 0)
else:
    s = img1.shape[0] / img2.shape[0]
    img2 = cv2.resize(img2, (int(s*img2.shape[1]), int(s*img2.shape[0])), 0)

if img3.shape[0] > img4.shape[0]:
    s = img3.shape[0] / img4.shape[0]
    img3 = cv2.resize(img3, (int(s*img3.shape[1]), int(s*img3.shape[0])), 0)
else:
    s = img3.shape[0] / img4.shape[0]
    img4 = cv2.resize(img4, (int(s*img4.shape[1]), int(s*img4.shape[0])), 0)

imgCombined1 = np.zeros((img1.shape[0], img1.shape[1]+img2.shape[1], 3), np.uint8)
imgCombined1[:, 0:img1.shape[1]] = img1
imgCombined1[:, img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
imgCombined2 = np.zeros((img3.shape[0], img3.shape[1]+img4.shape[1], 3), np.uint8)
imgCombined2[:, 0:img3.shape[1]] = img3
imgCombined2[:, img3.shape[1]:img3.shape[1]+img4.shape[1]] = img4
# imgCombined2 = np.zeros((img3.shape[0], 2*img3.shape[1], 3), np.uint8)
# imgCombined2[:, 0:img3.shape[1]] = img3
# imgCombined2[:, img3.shape[1]:2*img3.shape[1]] = img4

# s = 0.2
# rimgCombined1 = cv2.resize(imgCombined1, (int(s*imgCombined1.shape[1]), int(s*imgCombined1.shape[0])), 0)
# rimgCombined2 = cv2.resize(imgCombined2, (int(s*imgCombined2.shape[1]), int(s*imgCombined2.shape[0])), 0)

cv2.imwrite(f"exp{expNumber}.jpg", imgCombined1, [cv2.IMWRITE_JPEG_QUALITY, 50])
cv2.imwrite(f"exp{expNumber}_corrected.jpg", imgCombined2, [cv2.IMWRITE_JPEG_QUALITY, 50])

# cv2.imshow("combined image 2", rimgCombined1)
# cv2.waitKey(0)
# cv2.imshow("combined image 2", rimgCombined2)
# cv2.waitKey(0)

