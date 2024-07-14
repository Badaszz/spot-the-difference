# Important imports
from app import app
from flask import request, render_template
import os
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import numpy as np

# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'
app.config['EXISTNG_FILE'] = 'app/static/original'
app.config['GENERATED_FILE'] = 'app/static/generated'

# Route to home 
@app.route("/", methods=["GET", "POST"])
def index():
    # Execute if request is GET
    if request.method == "GET":
        return render_template("index.html")

    # Execute if request is POST
    if request.method == "POST":
        # Get uploaded images
        file_upload = request.files['file_upload']
        filename = file_upload.filename

        file_upload2 = request.files['file_upload2']
        filename2 = file_upload2.filename

        # Resize and save the uploaded image
        uploaded_image = Image.open(file_upload).resize((250, 160))
        uploaded_image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'uploaded_image.jpg'))

        # Resize and save the original image to ensure both uploaded and original match in size
        original_image = Image.open(file_upload2).resize((250, 160))
        original_image.save(os.path.join(app.config['EXISTNG_FILE'], 'original_image.jpg'))

        # Read uploaded and original images as arrays
        original_image = cv2.imread(os.path.join(app.config['EXISTNG_FILE'], 'original_image.jpg'))
        uploaded_image = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'uploaded_image.jpg'))

        # Convert images to grayscale
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

        # Calculate structural similarity
        (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
        diff = (diff * 255).astype("uint8")

        # Calculate threshold
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh_inv = np.invert(thresh.copy())

        # Dilation
        kernel = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(thresh, kernel, iterations=2)
        dilate_inv = np.invert(dilate)

        # Contours
        cnts1 = cv2.findContours(thresh_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = imutils.grab_contours(cnts1)

        # Draw contours on original and uploaded images
        for contour in cnts1:
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.rectangle(uploaded_image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Display images with rectangles highlighting differences
        img_height = original_image.shape[0]
        x = np.zeros((img_height, 10, 3), np.uint8)
        result1 = np.hstack((original_image, x, uploaded_image))

        # Second set of contours
        cnts2 = cv2.findContours(dilate_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)

        for contour in cnts2:
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.rectangle(uploaded_image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Display images with rectangles highlighting differences
        #result2 = np.hstack((original_image, x, uploaded_image))

        # Save generated images (if needed)
        cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_differences_1.jpg'), result1)
        cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_differences_2.jpg'), result1)

        # Calculate number of differences
        p = len(cnts2)
        c = 1 if p > 1 else 0  # Condition for plural

        # Return result to template
        return render_template('index.html', pred=f"{round(score * 100, 2)}% difficulty\n{p} difference{'s' * c}")

    return render_template('index.html')  # Default return for GET request

# Main function
if __name__ == '__main__':
    app.run(debug=True)
