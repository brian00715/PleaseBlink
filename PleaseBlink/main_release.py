"""
 # @ Author: Kenneth Simon
 # @ Email: smkk00715@gmail.com
 # @ Create Time: 2023-10-14 21:41:53
 # @ Modified time: 2023-10-15 01:21:11
 # @ Description: Release version for minize third-party dependencies
 """


import argparse
import base64
import os
import sys
import threading
import time

import cv2
import dlib
import grpc
import rumps
import numpy as np

import blink_detection_pb2
import blink_detection_pb2_grpc


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


def is_valid_ip_port(ip_port_str):
    if len(ip_port_str.split(":")) == 2:
        return True
    else:
        return False


def euclidean_dist(pt1, pt2):
    return np.sqrt(np.sum((pt1 - pt2) ** 2))


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


class StatusBarApp(rumps.App):
    def __init__(
        self,
        detect_mode="local",
        remote_host=("localhost", 12345),
    ):
        super(StatusBarApp, self).__init__("Eye Blink Monitor")
        self.menu = [
            "Enable",
            "Show detection frame",
            "Detection mode",
            "Set detection side",
            "🔧Set timeout threshold",
            "🔧Set detection sensitivity",
            "🔧Set detection frequency",
            "🔧Set blink count threshold",
            "🔧Set notification duty",
            "⛔️Quit",
        ]
        self.title = "Eye🟢"
        self.quit_button = None  # Hide the Quit button
        self.menu["Set detection side"].title = "🛫Change detection to remote"
        self.menu["Detection mode"].title = "👁️Detection mode: Local"

        self.detect_mode_flag = 1
        self.detect_mode = detect_mode
        remote_host_ar = remote_host.split(":")
        self.remote_host = (remote_host_ar[0], int(remote_host_ar[1]))
        self.show_frame_flag = -1

        self.timeout_th = 5
        self.ear_diff_th = [-0.03, -0.02]
        self.detect_freq = 60
        self.blink_cnt_th = 20
        self.noti_duty = 120
        self.main_enable = 0

        self.detection_sensitive = 0.5
        self.ear_diff_lb = -0.06
        self.qt_process = None

        # for eye detection----------------------------------------------------------------
        self.frame_draw = None
        self.blink_cnt = 0
        self.width = 600
        self.height = 720
        (self.lStart, self.lEnd) = (42, 48)
        (self.rStart, self.rEnd) = (36, 42)
        if os.getenv("RESOURCEPATH") is not None:
            self.dat_path = os.path.join(os.getenv("RESOURCEPATH"), "shape_predictor_68_face_landmarks.dat")
        else:
            self.dat_path = os.path.join(sys.path[0], "shape_predictor_68_face_landmarks.dat")

        if self.detect_mode == "local":
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.dat_path)
        elif self.detect_mode == "remote":
            self.channel = grpc.insecure_channel(f"{self.remote_host[0]}:{self.remote_host[1]}")
            try:
                grpc.channel_ready_future(self.channel).result(5)
            except:
                raise Exception("Connection to remote host timeout. You can try local mode later.")
            self.stub = blink_detection_pb2_grpc.BlinkDetectionStub(self.channel)

        self.ear_filter = LowPassFilter(0.5)
        self.ear_diff_list = []
        self.ear_list = []
        self.ear_last = 0
        self.last_blink_t = time.time()
        self.vs = None
        self.blink_detect_thread = threading.Thread(target=self.run_blink_detect)
        self.blink_detect_thread.start()

    @rumps.timer(1 / 30)
    def timer_test(self, _):
        if self.frame_draw is not None and self.show_frame_flag == 1:
            cv2.imshow("Detection Frame", self.frame_draw)
            cv2.waitKey(1)

    @rumps.clicked("Set detection side")
    def toggle_detect_side(self, sender):
        temp_main_enable = self.main_enable
        self.main_enable = 0  # disable the detector temporarily
        self.detect_mode_flag = -self.detect_mode_flag
        change_success = False
        if self.detect_mode_flag == -1:
            host_valid = False
            while not host_valid:
                response = rumps.Window(
                    "Enter remote host (ip:port):",
                    default_text=str("localhost:12345"),
                    dimensions=(150, 20),
                ).run()
                if response.clicked:
                    remote_host = response.text
                    if not is_valid_ip_port(remote_host):
                        rumps.alert("Error!", "Please enter a valid host!")
                        continue
                    remote_host_ar = remote_host.split(":")
                    self.remote_host = (remote_host_ar[0], int(remote_host_ar[1]))
                    host_valid = True
            change_success = self.change_detect_mode("remote", self.remote_host)
            if change_success:
                self.menu["Set detection side"].title = "🛬Change detection to local"
                self.menu["Detection mode"].title = "👁️Detection mode: Remote"
        elif self.detect_mode_flag == 1:  # remote
            change_success = self.change_detect_mode("local")
            if change_success:
                self.menu["Set detection side"].title = "🛫Change detection to remote"
                self.menu["Detection mode"].title = "👁️Detection mode: Local"
        if not change_success:
            self.detect_mode_flag = -self.detect_mode_flag  # change back
        if temp_main_enable != 0:  # if the detector was enabled before
            self.main_enable = 1  # enable the detector

    @rumps.clicked("Enable")
    def toggle_enable(self, sender):
        sender.state = not sender.state
        if sender.state:
            self.main_enable = 1
        else:
            self.main_enable = 0

    @rumps.clicked("🔧Set blink count threshold")
    def set_blink_cnt_th(self, _):
        response = rumps.Window(
            "Enter blink count threshold [15,50]:",
            default_text=str(self.blink_cnt_th),
            dimensions=(100, 25),
        ).run()
        if response.clicked:
            try:
                self.blink_cnt_th = int(response.text)
                print("Blink count threshold set to {} times".format(self.blink_cnt_th))
            except ValueError:
                rumps.alert("Please enter a valid number!")

    @rumps.clicked("🔧Set notification duty")
    def set_noti_duty(self, _):
        response = rumps.Window(
            "Enter notification duty in seconds:",
            default_text=str(self.noti_duty),
            dimensions=(100, 25),
        ).run()
        if response.clicked:
            try:
                self.noti_duty = int(response.text)
                print("Notification duty set to {} seconds".format(self.noti_duty))
            except ValueError:
                rumps.alert("Please enter a valid number!")

    @rumps.clicked("🔧Set detection frequency")
    def set_detect_freq(self, _):
        response = rumps.Window(
            "Enter detection frequency [1,60] Hz:",
            default_text=str(self.detect_freq),
            dimensions=(100, 25),
        ).run()
        if response.clicked:
            try:
                self.detect_freq = int(response.text)
                print("Detection frequency set to {} Hz".format(self.detect_freq))
            except ValueError:
                rumps.alert("Please enter a valid number!")

    @rumps.clicked("Show detection frame")
    def toggle_show_frame(self, sender):
        self.show_frame_flag = -self.show_frame_flag
        sender.state = not sender.state
        if self.show_frame_flag == -1:
            print("Hiding video frame")
            cv2.destroyAllWindows()

        elif self.show_frame_flag == 1:
            print("Showing video frame")

    @rumps.clicked("🔧Set timeout threshold")
    def set_timeout(self, _):
        response = rumps.Window(
            "Enter timeout threshold in seconds:",
            default_text=str(self.timeout_th),
            dimensions=(100, 25),
        ).run()
        if response.clicked:
            try:
                self.timeout_th = int(response.text)
                print("Timeout threshold set to {} seconds".format(self.timeout_th))
            except ValueError:
                rumps.alert("Please enter a valid number!")

    @rumps.clicked("🔧Set detection sensitivity")
    def set_sensitivity(self, _):
        response = rumps.Window(
            "Enter EAR difference threshold [0,1]:",
            default_text=str(self.detection_sensitive),
            dimensions=(100, 25),
        ).run()
        if response.clicked:
            try:
                self.detection_sensitive = float(response.text)
                th = self.sensitivity2th(self.detection_sensitive)
                self.ear_diff_th[0] = th[0]
                self.ear_diff_th[1] = th[1]
            except ValueError:
                rumps.alert("Please enter a valid number!")

    @rumps.clicked("⛔️Quit")
    def quit(self, _):
        self.show_frame_flag = -2
        cv2.destroyAllWindows()
        rumps.quit_application()

    def sensitivity2th(self, sensitivity):
        th_low = self.ear_diff_lb * sensitivity - self.ear_diff_lb
        offset = -0.02 * sensitivity + 0.02
        th_high = th_low + offset
        return [th_low, th_high]

    def set_icon(self, status):
        if status == 1:
            self.title = "Eye🟢"
        elif status == -1:
            self.title = "Eye🔴"

    def eye_aspect_ratio(self, eye):
        A = euclidean_dist(eye[1], eye[5])
        B = euclidean_dist(eye[2], eye[4])
        C = euclidean_dist(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def change_detect_mode(self, detect_mode, remote_host=[None, None]):
        if detect_mode == "local":
            try:
                self.detector = dlib.get_frontal_face_detector()
                self.dat_path = os.path.join(sys.path[0], "shape_predictor_68_face_landmarks.dat")
                self.predictor = dlib.shape_predictor(self.dat_path)
            except:
                # raise Exception("Failed to load local model. You can try remote mode later.")
                rumps.alert("Failed to load local model. You can try remote mode later.")
                return False
        elif detect_mode == "remote":
            self.channel = grpc.insecure_channel(f"{remote_host[0]}:{remote_host[1]}")
            try:
                grpc.channel_ready_future(self.channel).result(5)
            except:
                # raise Exception("Connection to remote host timeout. You can try local mode later.")
                rumps.alert("Connection to remote host timeout. You can try local mode later.")
                return False
            self.detector = None
            self.predictor = None
            self.stub = blink_detection_pb2_grpc.BlinkDetectionStub(self.channel)
        self.detect_mode = detect_mode
        return True

    def run_blink_detect(self):
        camera_opened_flag = False
        last_noti_t = time.time()
        while True:
            if not self.main_enable:
                if camera_opened_flag:
                    if self.vs is not None:
                        self.vs.release()
                    self.vs = None
                    camera_opened_flag = False
                time.sleep(1)
                continue
            elif not camera_opened_flag:
                self.vs = cv2.VideoCapture(0)
                camera_opened_flag = True

            time.sleep(1 / self.detect_freq)
            ret, frame = self.vs.read()
            wid_st_idx = int((frame.shape[1] - self.width) / 2)
            hei_st_idx = int((frame.shape[0] - self.height) / 2)
            frame = frame[hei_st_idx : hei_st_idx + self.height, wid_st_idx : wid_st_idx + self.width]
            frame = resize(frame, width=300)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ear = 0
            left_eye = []
            right_eye = []
            if self.detect_mode == "local":
                # detect faces in the grayscale frame
                rects = self.detector(gray, 0)
                # loop over the face detections
                for rect in rects:
                    shape = self.predictor(gray, rect)
                    shape = shape_to_np(shape)

                    # extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
                    left_eye = shape[self.lStart : self.lEnd]
                    right_eye = shape[self.rStart : self.rEnd]
                    left_EAR = self.eye_aspect_ratio(left_eye)
                    right_EAR = self.eye_aspect_ratio(right_eye)

                    # average the eye aspect ratio together for both eyes
                    ear = (left_EAR + right_EAR) / 2.0
                    ear = self.ear_filter(ear)

            elif self.detect_mode == "remote":
                # decode from base64 string
                frame_bin = cv2.imencode(".jpg", gray)[1]
                frame_str = base64.b64encode(frame_bin).decode("utf-8")
                request = blink_detection_pb2.GetEARRequest(image=frame_str)
                ear_response = self.stub.GetEAR(request)
                ear = ear_response.ear
                if ear == -1:
                    continue
                left_eye = np.array(ear_response.left_eye).reshape(-1, 2)
                right_eye = np.array(ear_response.right_eye).reshape(-1, 2)

            self.ear_list.append(ear)
            self.ear_diff = ear - self.ear_last
            self.ear_last = ear
            self.ear_diff_list.append(self.ear_diff)
            if len(self.ear_diff_list) > 3:
                self.ear_diff_list.pop(0)
            if len(self.ear_list) > 100:
                self.ear_list.pop(0)

            if time.time() - self.last_blink_t > self.timeout_th:
                self.set_icon(-1)
            else:
                self.set_icon(1)

            if self.ear_diff_list[0] < self.ear_diff_th[0] and self.ear_diff_list[-1] > self.ear_diff_th[1]:
                self.blink_cnt += 1
                self.last_blink_t = time.time()

            if self.show_frame_flag == 1:
                if not (len(left_eye) == 0 or len(right_eye) == 0):
                    # eye to cv2.umat
                    leftEyeHull = cv2.convexHull(left_eye)
                    rightEyeHull = cv2.convexHull(right_eye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.putText(
                    frame,
                    "Blinks: {}".format(self.blink_cnt),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "EAR_DIFF: {:>6.3f}".format(self.ear_diff),
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                self.frame_draw = frame.copy()

            if time.time() - last_noti_t > self.noti_duty:
                last_noti_t = time.time()
                if self.blink_cnt < (self.blink_cnt_th * self.noti_duty / 60):
                    rumps.notification(
                        "Alert",
                        "👁️ Please blink! 👁️",
                        f"You only blinked {self.blink_cnt} times in the last {self.noti_duty} seconds!",
                    )

                self.blink_cnt = 0
            if self.show_frame_flag == -2:
                break
        print("blink detector terminated")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--detect_mode", type=str, default="local", help="local or remote")
    arg_parser.add_argument("--remote_host", type=str, default="localhost:12345", help="remote host address")
    args = arg_parser.parse_args()

    status_icon = StatusBarApp(
        args.detect_mode,
        args.remote_host,
    )

    status_icon.run()
    status_icon.run_blink_detect_thread.join()
