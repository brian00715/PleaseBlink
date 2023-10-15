import argparse
import time

import cv2
import dlib
import imutils
import ipdb
import matplotlib.pyplot as plt
from imutils import face_utils
from imutils.video import FileVideoStream, VideoStream
from scipy.spatial import distance as dist


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


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


width = 600
height = 720

enable_plot = True
enable_viz = True

if __name__ == "__main__":
    blink_cnt = 0

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    print("[INFO] starting video stream thread...")
    print("[INFO] print q to quit...")
    vs = VideoStream(src=0).start()

    # time.sleep(1.0)
    ear_filter = LowPassFilter(0.5)
    ear_diff_list = []
    ear_list = []
    ear_last = 0
    last_blink_t = time.time()
    ear_diff_last = 0

    while True:
        frame = vs.read()
        wid_st_idx = int((frame.shape[1] - width) / 2)
        hei_st_idx = int((frame.shape[0] - height) / 2)
        frame = frame[hei_st_idx : hei_st_idx + height, wid_st_idx : wid_st_idx + width]
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        # loop over the face detections
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            ear = ear_filter(ear)
            ear_diff = ear - ear_last
            ear_last = ear
            ear_diff_list.append(ear_diff)
            if len(ear_diff_list) > 3:
                ear_diff_list.pop(0)
            if enable_plot:
                ear_list.append(ear_diff)
                if len(ear_list) > 50:
                    ear_list.pop(0)
                plt.clf()
                plt.plot(ear_list, "r")
                plt.pause(0.01)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if time.time() - last_blink_t > 5:  # not blinking
                # print("alert:", time.time())
                pass
            EAR_DIFF_TH_LB = -0.03
            EAR_DIFF_TH_UB = 0.0
            if ear_diff_list[0] < EAR_DIFF_TH_LB and ear_diff_list[-1] > EAR_DIFF_TH_LB:
                blink_cnt += 1
                last_blink_t = time.time()

            cv2.putText(frame, "Blinks: {}".format(blink_cnt), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if enable_viz:
            cv2.imshow("Frame", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
