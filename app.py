#!/usr/bin/env python3

from flask import Flask, jsonify, request
import numpy as np
import base64
import cv2

# set up the application
app = Flask(__name__)

# set up the face detector model
fmodel = cv2.dnn.readNetFromDarknet('../model-weights/yolov3-face.cfg',
                              '../model-weights/yolov3-wider_16000.weights')

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

@app.route('/detect', methods=['POST'])
def getData():
    # retrive the information sent
    recieved_data = request.form
    # get the images string
    imgStr = base64.b64decode(recieved_data['imgB'].encode('utf-8'))
    # get the extension
    ext = recieved_data['ext']
    # turn the image string into numpy array
    nparr = np.frombuffer(imgStr, np.uint8)
    # turn the numpy array into image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # create a blob from the data
    blob = cv2.dnn.blobFromImage(image=img, scalefactor=1/255,
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
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # encode the image into bytes
    _, resB = cv2.imencode(f'.{ext}', img)
    # encode the bytes into base64
    resB = base64.b64encode(resB).decode('utf-8')
    # setup the responce body
    res = {'detected_faces': resB}
    # return the response
    return jsonify(res), 200

# run the app
if __name__ == '__main__':
    app.run(debug=True)
