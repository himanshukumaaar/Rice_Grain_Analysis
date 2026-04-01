import cv2

def load_image(path):
    return cv2.imread(path)

def enhance_image(img, alpha=1.2, beta=30, ksize=(5,5)):
    enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    blurred = cv2.GaussianBlur(enhanced, ksize, 0)
    return blurred
