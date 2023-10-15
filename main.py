"""
 # @ Author: Kenneth Simon
 # @ Email: smkk00715@gmail.com
 # @ Create Time: 2023-10-14 21:41:53
 # @ Modified time: 2023-10-15 01:21:11
 # @ Description: This is a script to alert the user through status menu if he is not blinking 
 # for a long time.
 """


import threading
import time
from multiprocessing import Array, Process, Queue, Value

import cv2
import dlib
import imutils
import matplotlib.pyplot as plt
import rumps
from imutils import face_utils
from imutils.video import VideoStream
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QSlider, QVBoxLayout, QWidget
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


class StatusBarIcon(rumps.App):
    def __init__(
        self,
        show_frame_flag: Value,
        status_icon_flag: Value,
        time_out_th: Value,
        ear_diff_th: Array,
        frame_queue: Queue,
    ):
        super(StatusBarIcon, self).__init__("Eye Blink Monitor")
        # self.icon = "green.svg"  # Default icon
        self.menu = ["Toggle cv2.imshow", "Set Timeout Threshold", "Set Detection Sensitivity", "QuitAll"]
        self.menu["Toggle cv2.imshow"].title = "Open cv2.imshow"
        self.quit_button = None  # Hide the Quit button

        self.show_frame_flag = show_frame_flag
        self.status_icon_flag = status_icon_flag
        self.frame_queue = frame_queue
        self.time_out_th = time_out_th
        self.detection_sensitive = 0.5
        self.ear_diff_th = ear_diff_th
        self.ear_diff_lb = -0.06
        self.qt_process = None

    def set_icon(self, status):
        if status == 1:
            self.title = "🟢"
        elif status == -1:
            self.title = "🔴"

    @rumps.clicked("Toggle cv2.imshow")
    def toggle_show_frame(self, _):
        self.show_frame_flag.value = -self.show_frame_flag.value
        if self.show_frame_flag.value == -1:
            self.menu["Toggle cv2.imshow"].title = "Open cv2.imshow"
            print("Hiding video frame")
            if not self.qt_process is None:
                self.qt_process.terminate()
                self.qt_process = None
        elif self.show_frame_flag.value == 1:
            self.menu["Toggle cv2.imshow"].title = "Close cv2.imshow"
            print("Showing video frame")
            self.qt_process = Process(
                target=show_video_frame, args=(self.frame_queue, self.show_frame_flag, self.ear_diff_th)
            )
            self.qt_process.start()

    @rumps.clicked("Set Timeout Threshold")
    def set_timeout(self, _):
        response = rumps.Window("Enter timeout threshold in seconds:", default_text=str(self.time_out_th.value)).run()
        if response.clicked:
            try:
                self.time_out_th.value = int(response.text)
                print("Timeout threshold set to {} seconds".format(self.time_out_th.value))
            except ValueError:
                rumps.alert("Please enter a valid number!")

    @rumps.clicked("QuitAll")
    def quit(self, _):
        self.show_frame_flag.value = -2
        time.sleep(1)
        rumps.quit_application()

    @rumps.clicked("Set Detection Sensitivity")
    def set_sensitivity(self, _):
        response = rumps.Window(
            "Enter EAR difference threshold [0,1]:", default_text=str(self.detection_sensitive)
        ).run()
        if response.clicked:
            try:
                self.detection_sensitive = float(response.text)
                th = self.sensitivity2th(self.detection_sensitive)
                self.ear_diff_th[0] = th[0]
                self.ear_diff_th[1] = th[1]
            except ValueError:
                rumps.alert("Please enter a valid number!")

    def sensitivity2th(self, sensitivity):
        th_low = self.ear_diff_lb * sensitivity - self.ear_diff_lb
        offset = -0.02 * sensitivity + 0.02
        th_high = th_low + offset
        return [th_low, th_high]


def show_video_frame(frame_queue: Queue, show_frame_flag: Value, ear_diff_th: Array):
    app = QApplication([])
    window = QWidget()
    layout = QVBoxLayout()
    label = QLabel()
    ear_diff_lb_label = QLabel("EAR_DIFF_LB: {:.3f}".format(ear_diff_th[0]))
    ear_diff_ub_label = QLabel("EAR_DIFF_UB: {:.3f}".format(ear_diff_th[1]))
    scale_factor = 1000

    def set_ear_diff_th_lb(val):
        ear_diff_th[0] = val / scale_factor
        # print("EAR difference lower bound set to: ", ear_diff_th[0])
        ear_diff_lb_label.setText("EAR_DIFF_LB: {:.3f}".format(ear_diff_th[0]))

    def set_ear_diff_th_ub(val):
        ear_diff_th[1] = val / scale_factor
        # print("EAR difference upper bound set to: ", ear_diff_th[1])
        ear_diff_ub_label.setText("EAR_DIFF_UB: {:.3f}".format(ear_diff_th[1]))

    # add slider to control the threshold
    ear_diff_th_lb_slider = QSlider(Qt.Orientation.Horizontal)
    ear_diff_th_lb_slider.setMinimum(-50)
    ear_diff_th_lb_slider.setMaximum(0)
    ear_diff_th_lb_slider.setValue(int(ear_diff_th[0] * scale_factor))
    ear_diff_th_lb_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    ear_diff_th_lb_slider.setTickInterval(1)
    ear_diff_th_lb_slider.valueChanged.connect(set_ear_diff_th_lb)
    layout.addWidget(ear_diff_lb_label)
    layout.addWidget(ear_diff_th_lb_slider)

    ear_diff_th_ub_slider = QSlider(Qt.Orientation.Horizontal)
    ear_diff_th_ub_slider.setMinimum(-30)
    ear_diff_th_ub_slider.setMaximum(20)
    ear_diff_th_ub_slider.setValue(int(ear_diff_th[1] * scale_factor))
    ear_diff_th_ub_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    ear_diff_th_ub_slider.setTickInterval(1)
    ear_diff_th_ub_slider.valueChanged.connect(set_ear_diff_th_ub)
    layout.addWidget(ear_diff_ub_label)
    layout.addWidget(ear_diff_th_ub_slider)

    layout.addWidget(label)
    window.setLayout(layout)

    def update_frame():
        if show_frame_flag.value == -1:
            window.hide()
        elif show_frame_flag.value == 1:
            window.show()
        elif show_frame_flag.value == -2:
            while not frame_queue.empty():
                frame_queue.get()  # clear the queue
            timer.stop()
            window.close()
            print("Qt process terminated")
            app.quit()

        if not frame_queue.empty():
            frame = frame_queue.get()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(pixmap)

    timer = QTimer()
    timer.timeout.connect(update_frame)
    timer.start(int(1 / 30 * 1000))
    window.show()
    app.exec()


class BlinkDetector:
    def __init__(
        self,
        frame_queue: Queue,
        show_frame_flag: Value,
        status_icon_flag: Value,
        ear_diff_th: Array,
        timeout_th: Value,
        set_icon_func,
        detect_freq,
    ):
        self.frame_queue = frame_queue
        self.show_frame_flag = show_frame_flag
        self.status_icon_flag = status_icon_flag
        self.ear_diff_th = ear_diff_th
        self.set_icon = set_icon_func
        self.time_out_th = timeout_th
        self.detect_freq = detect_freq

        self.width = 600
        self.height = 720

        self.blink_cnt = 0
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        self.vs = VideoStream(src=0).start()
        self.ear_filter = LowPassFilter(0.5)
        self.ear_diff_list = []
        self.ear_list = []
        self.ear_last = 0
        self.last_blink_t = time.time()

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def run(self):
        while True:
            time.sleep(1 / self.detect_freq.value)
            frame = self.vs.read()
            wid_st_idx = int((frame.shape[1] - self.width) / 2)
            hei_st_idx = int((frame.shape[0] - self.height) / 2)
            frame = frame[hei_st_idx : hei_st_idx + self.height, wid_st_idx : wid_st_idx + self.width]
            frame = imutils.resize(frame, width=300)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)
            # loop over the face detections
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[self.lStart : self.lEnd]
                rightEye = shape[self.rStart : self.rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                ear = self.ear_filter(ear)
                self.ear_list.append(ear)
                self.ear_diff = ear - self.ear_last
                self.ear_last = ear
                self.ear_diff_list.append(self.ear_diff)
                if len(self.ear_diff_list) > 3:
                    self.ear_diff_list.pop(0)
                if len(self.ear_list) > 100:
                    self.ear_list.pop(0)

                if 0:
                    plt.plot(self.ear_list)
                    # plt.plot(self.ear_diff_list)
                    plt.show()
                    plt.pause(0.001)

                if time.time() - self.last_blink_t > self.time_out_th.value:
                    self.set_icon(-1)
                else:
                    self.set_icon(1)
                    rumps.notification(
                        "Alert❗️", "👁️ Please blink! 👁️", "You have been staring at the screen for too long!"
                    )

                if self.ear_diff_list[0] < self.ear_diff_th[0] and self.ear_diff_list[-1] > self.ear_diff_th[1]:
                    self.blink_cnt += 1
                    self.last_blink_t = time.time()

                if self.show_frame_flag.value == 1:
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
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
                    cv2.putText(
                        frame, "EAR: {:.2f}".format(ear), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )
                    cv2.putText(
                        frame,
                        "EAR_DIFF: {:>6.3f}".format(self.ear_diff),
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    if self.frame_queue.empty():
                        frame_queue.put(frame)
            if self.show_frame_flag.value == -2:
                break
        frame_queue.close()
        print("blink detector terminated")


if __name__ == "__main__":
    # shared variables between processes
    frame_queue = Queue()
    show_frame_flag = Value("i", -1)
    status_icon_flag = Value("i", -1)
    timeout_th = Value("i", 5)
    ear_diff_th = Array("f", [-0.03, -0.02])
    detect_freq = Value("i", 30)

    status_icon = StatusBarIcon(show_frame_flag, status_icon_flag, timeout_th, ear_diff_th, frame_queue)

    blink_detector = BlinkDetector(
        frame_queue,
        show_frame_flag,
        status_icon_flag,
        ear_diff_th,
        timeout_th,
        status_icon.set_icon,
        detect_freq,
    )
    blink_thread = threading.Thread(target=blink_detector.run)
    blink_thread.start()

    status_icon.run()
    blink_thread.join()
