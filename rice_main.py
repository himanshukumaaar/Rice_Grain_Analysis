import os
import cv2
from rice_detect import detect_and_extract
from rice_dataframe import create_df, classify, save_to_csv


def load_image(path):
    return cv2.imread(path)

# Generator function to yield one image at a time from nested folders
def load_images_generator(base_dir='Images/Rice_Image_Dataset'):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                img = load_image(image_path)
                if img is not None:
                    yield (image_path, img)

# Example processing loop for large dataset
image_generator = load_images_generator()

count = 0
os.makedirs("results", exist_ok=True)

for image_path, img in image_generator:
    print(f"Processing image {count+1}: {image_path}")

    # Preprocess: grayscale + threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Detect features
    features, contours = detect_and_extract(binary, img)

    # Create DataFrame + classify
    df = create_df(features)
    df = classify(df)

    # Save results to CSV
    save_to_csv(df, f"results/result_{count+1}.csv")

    # Draw contours and IDs
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # Save processed image
    cv2.imwrite(f"results/processed_{count+1}.jpg", img)

    # Show only first image for verification
    if count == 0:
        cv2.imshow("Processed First Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    count += 1

print(f"✅ Total images processed: {count}")




































