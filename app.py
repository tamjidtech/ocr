from flask import Flask, request, jsonify
import cv2
import pytesseract
import numpy as np

app = Flask(__name__)

@app.route("/")
def welcone():
    return 'App is Running...'

@app.route("/img", methods=["POST"])
def process_image():
    pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
    file = request.files['image']
    img = cv2.imdecode(np.asarray(bytearray(file.stream.read()), dtype=np.uint8), cv2.IMREAD_COLOR)

    result = img.copy()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_bw = cv2.threshold(grey, 150, 230, cv2.THRESH_TOZERO)
    
    def noise_removal(image):
        kernel = np.ones((1,1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1,1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 1)
        return image

    def thick_font(image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((1,1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image


    blur = cv2.GaussianBlur(grey, (3,3), 0)
    thresh= cv2.threshold(blur,0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,7))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    num = 0
    cnts = [(0, 0, 650, 300),(510, 0, 700, 200)]
    for c in cnts:
        x,y,w,h = c   
        roi = img[y:y+h, x:x+h]
        if num == 0:
            grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            threshold, im_bw = cv2.threshold(grey, 150, 230, cv2.THRESH_TOZERO)
            removal = noise_removal(im_bw)        
            thick = thick_font(removal)
            medic = pytesseract.image_to_string(thick, )
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            # Remove horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45,1))
            remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(result, [c], -1, (255,255,255), 5)

            # Remove vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
            remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(result, [c], -1, (255,255,255), 5)
            
            roi = result[y:y+h, x:x+h]
            custom_oem=r'digits --oem 1 --psm 7 -c tessedit_char_whitelist=0123456789'
            digit = pytesseract.image_to_string(roi, config= custom_oem)            

        num =+1

    dec_med = medic.strip().split('\n')
    dec_med = list(filter(str.strip, dec_med))
    dec_qty = digit.strip().split('\n')
    return jsonify({'dec_med': dec_med, 'dec_qty': dec_qty, 'medic': medic, 'digit': digit})
