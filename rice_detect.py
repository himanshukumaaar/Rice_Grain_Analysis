import cv2
import numpy as np

def detect_and_extract(processed_img, original_img):
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # create mask for current contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # mean color within mask
        mean_val = cv2.mean(original_img, mask=mask)  # (B, G, R, A)
        avg_b, avg_g, avg_r = mean_val[:3]

        features.append({
            'Grain_ID': i + 1,
            'Area': int(area),
            'Length': int(max(w, h)),
            'Width': int(min(w, h)),
            'Avg_R': round(avg_r, 2),
            'Avg_G': round(avg_g, 2),
            'Avg_B': round(avg_b, 2)
        })

    return features, contours