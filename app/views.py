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

	# Execute if request is get
	if request.method == "GET":
	    return render_template("index.html")

	# Execute if reuqest is post
	if request.method == "POST":
                # Get uploaded image
                file_upload = request.files['file_upload']
                filename = file_upload.filename
                
                # Resize and save the uploaded image
                uploaded_image = Image.open(file_upload).resize((250,160))
                uploaded_image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'uploaded_image.jpg'))

                # Resize and save the original image to ensure both uploaded and original matches in size
                original_image = Image.open(file_upload2).resize((250,160))
                original_image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'original_image.jpg'))

                # Read uploaded and original image as array
                original_image = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image2.jpg'))
                uploaded_image = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image1.jpg'))

                #extra images for second parameter 
                img3 = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image2.jpg'))
                img4 = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image1.jpg'))

                #setting height 
                img_height = original_image.shape[0]

                # Convert image into grayscale
                original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

                # Calculate structural similarity
                (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
                diff = (diff * 255).astype("uint8")

                # Calculate threshold 
                thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                thresh_inv = np.invert(thresh.copy())

                #Dilation 
                kernel = np.ones((5,5), np.uint8) 
                dilate = cv2.dilate(thresh, kernel, iterations=2) 
                cv2.imshow("Dilate", dilate)
                dilate_inv = np.invert(dilate)

                #contours
                cnts1 = cv2.findContours(thresh_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts1 = imutils.grab_contours(cnts1)
                
                #second contours
                cnts2 = cv2.findContours(dilate_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts2 = imutils.grab_contours(cnts2)

                # Draw contours on image
                for contour in cnts1:
                    if cv2.contourArea(contour) > 10:
                        # Calculate bounding box around contour
                        x, y, w, h = cv2.boundingRect(contour)
                        # Draw rectangle - bounding box on both images
                        cv2.rectangle(original_image, (x, y), (x+w, y+h), (255,0,0), 1)
                        cv2.rectangle(uploaded_image, (x, y), (x+w, y+h), (255,0,0), 1)

                # Show images with rectangles on differences
                x = np.zeros((img_height,10,3), np.uint8)
                result1 = np.hstack((ioriginal_image, x, uploaded_image))

                #draw contours on image


                for contour in cnts2:
                    if cv2.contourArea(contour) > 10:
                        # Calculate bounding box around contour
                        x, y, w, h = cv2.boundingRect(contour)
                        # Draw rectangle - bounding box on both images
                        cv2.rectangle(img3, (x, y), (x+w, y+h), (255,0,0), 1)
                        cv2.rectangle(img4, (x, y), (x+w, y+h), (255,0,0), 1)
                        p += 1

                # Show images with rectangles on differences
                x = np.zeros((img_height,10,3), np.uint8)
                result1 = np.hstack((img3, x, img4))

                #conditions for plural 
                c=0
                if p>1: c = 1


                # Save all output images (if required)
                #cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_original.jpg'), original_image)
                #cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_uploaded.jpg'), uploaded_image)
                cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_differences_1.jpg'), result1)
                cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_differences_2.jpg'), result2)
                return render_template('index.html',pred=str(round(score*100,2)) + '%' + ' difficulty' + '\n' + p + 'difference' + 's'* c )

       
# Main function
if __name__ == '__main__':
    app.run(debug=True)
