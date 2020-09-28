from threading import Thread
import cv2
import os
import shutil
import numpy as np
import smtplib
import secrets
import urllib.request
from email.message import EmailMessage


def load_image(path, size=None):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size:
        image = cv2.resize(image, size)
    return image.astype(np.uint8)


def load_url_image(url, size=None):
    img_request = urllib.request.urlopen(url)
    image = np.array(bytearray(img_request.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    if size:
        image = cv2.resize(image, size)
    return image


def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row, content in enumerate(lines):
            labels[row] = {"id": row, "name": content.strip()}
    return labels


def preprocess_input(image):
    processed = (2.0 / 255.0) * image - 1.0
    processed = np.expand_dims(processed, axis=0)
    return processed.astype(np.float32)


def get_category_index(scores, classes, min_threshold=0.5):
    for i in range(len(scores)):
        if (scores[i] > min_threshold) and (scores[i] <= 1.0):
            return classes[i]
    return 0


def draw_boxes(image, boxes, classes, scores,
               category_index, height, width, min_threshold=0.5):
    for i in range(len(scores)):
        if (scores[i] > min_threshold) and (scores[i] <= 1.0):
            y_min = int(max(1, (boxes[i][0] * height)))
            x_min = int(max(1, (boxes[i][1] * width)))
            y_max = int(min(height, (boxes[i][2] * height)))
            x_max = int(min(width, (boxes[i][3] * width)))

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (10, 255, 0), 2)
            if classes[i] in category_index:
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'N/A'
            label = f'{class_name}: {int(scores[i] * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label,
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_min_y = max(y_min, labelSize[1] + 10)
            cv2.rectangle(image, (x_min, label_min_y - labelSize[1] - 10),
                          (x_min + labelSize[0], label_min_y + baseLine - 10),
                          (255, 255, 255),
                          cv2.FILLED)
            cv2.putText(image, label, (x_min, label_min_y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return image


class VideoStream:

    def __init__(self, resolution=(640, 480), cam=0):
        self.stream = cv2.VideoCapture(cam)
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class IPStream:

    def __init__(self, stream_url, resolution=(640, 480)):
        self.stream = stream_url
        self.resolution = resolution

        # Read first frame from the stream
        self.frame = load_url_image(stream_url, resolution)

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                return

            # Otherwise, grab the next frame from the stream
            self.frame = load_url_image(self.stream, self.resolution)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class SaveVideo:

    def __init__(self, frames_path, output_path, fps, resolution=(640, 480)):
        self.codec = cv2.VideoWriter_fourcc(*"XVID")
        self.frames_path = frames_path
        self.output_path = os.path.join(output_path, f"{os.path.basename(frames_path)}.avi")
        self.resolution = resolution
        self.fps = fps

    def save(self):
        Thread(target=self.write, args=()).start()
        return self

    def write(self):
        out = cv2.VideoWriter(self.output_path, self.codec, self.fps, self.resolution)
        for frame in os.listdir(self.frames_path):
            frame_path = os.path.join(self.frames_path, frame)
            img_frame = cv2.imread(frame_path)
            out.write(img_frame)
        shutil.rmtree(self.frames_path)
        print(f"...Video file generated at {self.output_path}...")
        out.release()


class SendMail:

    def __init__(self, mail_id, password, recipients, **kwargs):
        super(SendMail, self).__init__(**kwargs)
        self.id = mail_id
        self.pwd = password
        self.recipients = recipients

    def send_mail(self, message, image):
        Thread(target=self.mail, args=(message, image)).start()

    def mail(self, message, image):
        msg = EmailMessage()
        msg["Subject"] = "An Intruder has been detected please check need Attention!"
        msg["From"] = self.id
        msg["To"] = self.recipients
        msg.set_content(message)
        msg.add_attachment(image, maintype="image",
                           subtype="jpeg", filename=f"{secrets.token_hex(8)}.jpg")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(self.id, self.pwd)
            smtp.send_message(msg)
        print(f"...Mail send successfully...")


def recover_videos(temp_path, output_path, fps, resolution=(640, 480)):
    codec = cv2.VideoWriter_fourcc(*"XVID")
    for frames_path in os.listdir(temp_path):
        output_path = os.path.join(output_path, f"{os.path.basename(frames_path)}.avi")
        out = cv2.VideoWriter(output_path, codec, fps, resolution)
        folder = os.path.join(temp_path, frames_path)
        for frame in os.listdir(folder):
            frame_path = os.path.join(folder, frame)
            img_frame = cv2.imread(frame_path)
            out.write(img_frame)
        shutil.rmtree(folder)
        out.release()


def create_mail_msg(class_name):
    return f"""HEY! An Intruder has been detected. Need your attention,It seems like a {class_name}"""


def get_image_bytes(image):
    is_success, im_buf_arr = cv2.imencode(".jpg", image)
    return im_buf_arr.tobytes()
