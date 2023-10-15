import blink_detection_pb2_grpc
import blink_detection_pb2
import grpc
import cv2
import base64
import numpy as np

if 1:
    channel = grpc.insecure_channel("127.0.0.1:12345")
    client = blink_detection_pb2_grpc.BlinkDetectionStub(channel)

    # request = blink_detection_pb2.Test(test="world")
    # response = client.SayHi(request)

    if 0:
        image = cv2.imread("frame.jpg")
        image_bin = cv2.imencode(".png", image)[1].tobytes()
        image_str = base64.b64encode(image_bin).decode("utf-8")
        request = blink_detection_pb2.GetEARRequest(image=image_str)
        ear_response = client.GetEAR(request)
        print("EAR:", ear_response.ear)
    if 1:
        vc = cv2.VideoCapture(0)
        while True:
            ret, frame = vc.read()
            if not ret:
                break
            _, image_bin = cv2.imencode(".jpg", frame)
            image_str = base64.b64encode(image_bin).decode("utf-8")
            request = blink_detection_pb2.GetEARRequest(image=image_str)
            ear_response = client.GetEAR(request)
            ear = ear_response.ear
            if ear == -1:
                continue
            left_eye = np.array(ear_response.left_eye).reshape(-1, 2)
            right_eye = np.array(ear_response.right_eye).reshape(-1, 2)
            # print("left_eye:", left_eye)
            # print("right_eye:", right_eye)
            leftEyeHull = cv2.convexHull(left_eye)
            rightEyeHull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
