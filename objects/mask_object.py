import cv2
import numpy as np
import os

import tflite_runtime.interpreter as tflite


class MaskObject(object):

    def __init__(self):
        print("[INFO] loading face detector model...")
        prototxt_path = os.path.expanduser("/home/pi/MaskDetection/models/face_detector/deploy.prototxt.txt")
        weights_path = os.path.expanduser(
            "/home/pi/MaskDetection/models/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
        self.face_net = cv2.dnn.readNet(prototxt_path, weights_path)

        print("[INFO] loading face mask detector model...")
        self.interpreter = tflite.Interpreter(
            model_path=os.path.expanduser("/home/pi/MaskDetection/models/mask_detector/model_quant.tflite"))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @staticmethod
    def calculate_box_position(index, width, height, detections, frame, faces, locs):
        # compute the (x, y)-coordinates of the bounding box for
        # the object
        box = detections[0, 0, index, 3:7] * np.array([width, height, width, height])
        (start_x, start_y, end_x, end_y) = box.astype("int")

        # ensure the bounding boxes fall within the dimensions of
        # the frame
        (start_x, start_y) = (max(0, start_x), max(0, start_y))
        (end_x, end_y) = (min(width - 1, end_x), min(height - 1, end_y))

        try:
            # extract the face ROI
            # start_x, start_y: top left corner
            # end_x, end_y: bottom right corner
            face = frame[start_y:end_y, start_x:end_x]

            # resize it to 160x160
            face = cv2.resize(face, (160, 160))

            # expand array shape from [160, 160, 3] to [1, 160, 160, 3]
            face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((start_y, start_x, end_y, end_x))
        except:
            pass

    def detect_faces(self, frame):
        # grab the dimensions of the frame and then construct a blob from it
        (height, wight) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (200, 200),
                                     (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        locs = []
        faces = []

        # loop over the detections
        for index in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, index, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                self.calculate_box_position(index, wight, height, detections, frame, faces, locs)
        return locs, faces

    def predict(self, faces):
        '''tflite'''
        labels = []
        scores = []
        for face in faces:
            # pre-process image to conform to MobileNetV2
            # input_mean = input_std = float(127.5)
            # input_data = (np.float32(face) - input_mean) / input_std

            # set our input tensor to our face image
            self.interpreter.set_tensor(self.input_details[0]['index'], np.float32(face))

            # perform classification
            self.interpreter.invoke()

            # get our output results tensor
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            result = np.squeeze(output_data)

            # get label from the result.
            # the class with the higher confidence is the label.
            (mask, withoutMask) = result
            label = "Mask" if mask > withoutMask else "No Mask"

            # get the highest confidence as the label's score
            score = np.max(result)

            labels.append(label)
            scores.append(score)
        return (labels, scores)
