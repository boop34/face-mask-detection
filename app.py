#!/usr/bin/env python3

from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, jsonify, request
from ResNet50 import ResNet50
import tensorflow as tf
import numpy as np
import json
import base64
import cv2
import os

# suppress the debuging info of tensorflow
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# set up the application
app = Flask(__name__)

# set up the face detector model
fmodel = cv2.dnn.readNetFromDarknet('model-weights/yolov3-face.cfg',
                                   'model-weights/yolov3-wider_16000.weights')

# set up the mask detector model
model = ResNet50(input_shape=(224, 224, 3), classes=3)
# load the weights
model.load_weights('model.h5')

# define the labels
labels = ['CWM', 'IWM', 'NWM']

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
        colRed = (0, 0, 255)
        colYellow = (0, 255, 255)
        colGreen = (0, 255, 0)
        colWhite = (255, 255, 255)

        # extract the face from the image
        tempImg = img[top: bottom, left: right]
        # resize it to 224x224 and process it to fit to the model
        tempImg = cv2.resize(tempImg, (224, 224))
        # swap red and blue channels for model to work
        tempImg = cv2.cvtColor(tempImg, cv2.COLOR_BGR2RGB)
        # prepare the image to fit the model
        tempImg = post_process_image(tempImg)

        # get the prediction
        prediction = model.predict(tempImg)
        # determine the label
        labelVal = np.argmax(prediction)
        label = labels[labelVal]
        # get the confidence from the prediction
        conf = prediction[0][labelVal]

        # set up the color accotding to the prediction
        if labelVal == 0:
            col = colGreen
        elif labelVal == 1:
            col = colYellow
        else:
            col = colRed

        # form the text to draw
        text = f'{label} - {conf * 100: .2f}%'
        # set up the text
        label_size, base_line = cv2.getTextSize(text,
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, 1)
        # draw the rectangle
        # img = cv2.rectangle(img, (left, top), (right, bottom), col, 2)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
        top = max(y, label_size[1])
        # draw the text
        img = cv2.putText(img, text, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, colWhite, 2)

    # convert the image to show it via PIL on the client end
    tempImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert it to list to send to client and return it
    return (tempImg.tolist(), label, conf)


@app.route('/detect', methods=['POST'])
def getData():
    # get the list of images
    imgList = json.loads(request.form['imgB'])

    # set up the response list
    resList = []
    for imgEnc, fname, ext in imgList:
        resImg, label, conf = detect_faces(imgEnc)
        oFname = f'{fname}_{label}_{conf * 100: .2f}'
        resList.append((resImg, oFname, ext))
    # setup the responce body
    res = {'detected_faces': resList}
    # return the response
    return jsonify(res), 200


# run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
