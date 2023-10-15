from concurrent import futures

import cv2
import dlib
import grpc
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist

import blink_detection_pb2
import blink_detection_pb2_grpc

import numpy as np
import base64


class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def __call__(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value
        return self.value


class BlinkDetectionServicer(blink_detection_pb2_grpc.BlinkDetectionServicer):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.width = 600
        self.height = 720
        self.ear_filter = LowPassFilter(0.5)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def GetEAR(self, request: blink_detection_pb2.GetEARRequest, context):
        # decode from base64 string
        frame = base64.b64decode(request.image)
        frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), -1)
        gray = frame
        # wid_st_idx = int((frame.shape[1] - self.width) / 2)
        # hei_st_idx = int((frame.shape[0] - self.height) / 2)
        # frame = frame[hei_st_idx : hei_st_idx + self.height, wid_st_idx : wid_st_idx + self.width]
        # frame = imutils.resize(frame, width=300)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = self.detector(gray, 0)
        response = None
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
            left_eye = shape[self.lStart : self.lEnd]
            right_eye = shape[self.rStart : self.rEnd]
            left_EAR = self.eye_aspect_ratio(left_eye)
            right_EAR = self.eye_aspect_ratio(right_eye)

            # average the eye aspect ratio together for both eyes
            ear = (left_EAR + right_EAR) / 2.0
            ear = self.ear_filter(ear)
            response = blink_detection_pb2.GetEARResponse()
            response.ear = ear
            response.left_eye.extend(left_eye.flatten())
            response.right_eye.extend(right_eye.flatten())
        if response is None:
            response = blink_detection_pb2.GetEARResponse()
            response.ear = -1
            response.left_eye.extend([-1])
            response.right_eye.extend([-1])
        return response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    blink_detection_pb2_grpc.add_BlinkDetectionServicer_to_server(BlinkDetectionServicer(), server)
    server.add_insecure_port("0.0.0.0:12345")
    server.start()
    print("Server started")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
