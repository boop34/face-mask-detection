#!/usr/bin/env python3

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import base64
import cv2

# supress the warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# set up the application
app = Flask(__name__)

# set up the face detector model
fmodel = cv2.dnn.readNetFromDarknet('model-weights/yolov3-face.cfg',
                                    'model-weights/yolov3-wider_16000.weights')

# set up the mask detector model
model = load_model('model.h5')

# define the labels
labels = ['cwm', 'iwm', 'nwm']

# set up a few helper function


def get_outputs_names(net):
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# function to extract the faces with most confidence
def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep
    # the ones with high confidence scores. Assign the box's class label as
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
    return final_boxes


# make the boxes a little bit bigger
def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 \
        else left - margin - 1

    right = right + margin

    return left, top, right, bottom


# process image to feed it to model
def post_process_image(img):
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# detect faces and labels
def detect_faces(imgEnc):
    # get the images string
    imgStr = base64.b64decode(imgEnc.encode('utf-8'))

    # turn the image string into numpy array
    nparr = np.frombuffer(imgStr, np.uint8)
    # turn the numpy array into image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # create a blob from the data
    blob = cv2.dnn.blobFromImage(image=img, scalefactor=1 / 255,
                                 size=(416, 416), mean=[0, 0, 0],
                                 swapRB=1, crop=False)

    # set the blob as the imput to the face detector model
    fmodel.setInput(blob)
    # do a forward pass to get all the possible faces
    outs = fmodel.forward(get_outputs_names(fmodel))
    # get all the faces with most confidence
    faces = post_process(img, outs, 0.5, 0.4)

    # draw the rectangle(s)
    for (x, y, w, h) in faces:
        # get the left, top, right, bottom coordinates
        left, top, right, bottom = refined_box(x, y, w, h)

        # define some colors
        colRed = (255, 0, 0)
        colYellow = (0, 255, 255)
        colGreen = (0, 255, 255)
        colWhite = (255, 255, 255)

        # extract the face from the image
        tempImg = img[top:bottom, left:right]
        # resize it to 224x224 and process it to fit to the model
        tempImg = cv2.resize(tempImg, (224, 224))
        tempImg = post_process_image(tempImg)

        # get the prediction
        prediction = model.predict(tempImg)
        # determine the label
        label = labels[np.argmax(prediction)]
        # get the confidence from the prediction
        conf = prediction[label]

        # set up the color accotding to the prediction
        if label == 'cwm':
            col = colGreen
        elif label == 'iwm':
            col = colYellow
        else:
            col = colRed

        # form the text to draw
        text = f'{label}-{conf: .3f}%'
        # set up the text
        label_size, base_line = cv2.getTextSize(text,
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, 1)
        # draw the rectangle
        img = cv2.rectangle(img, (left, top), (right, bottom), col, 2)
        top = max(top, label_size[1])
        # draw the text
        img = cv2.putText(img, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX,
                          0.4, colWhite, 1)

    # convert the image to show it via PIL on the client end
    tempImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert it to list to send to client and return it
    return tempImg.tolist()


@app.route('/detect', methods=['POST'])
def getData():
    # get the list of images
    imgList = request.form['imgB']

    # set up the response list
    resList = []
    for (imgEnc, fname, ext) in imgList:
        resImg = detect_faces(imgEnc)
        resList.append((resImg, fname, ext))
    # setup the responce body
    res = {'detected_faces': resList}
    # return the response
    return jsonify(res), 200


# run the app
if __name__ == '__main__':
    app.run(debug=True)
