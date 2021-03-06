import os
import time
import cv2
import secrets
import tensorflow as tf
import numpy as np
from utils.category import theft_category_index
from absl import app, flags, logging
from absl.flags import FLAGS
from utils.parse_args import get_webcam_arguments
from utils.utility import (draw_boxes, VideoStream, create_mail_msg, recover_videos,
                           get_category_index, SaveVideo, SendMail, get_image_bytes)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

flags.DEFINE_string('config', "data/configs/theft.config",
                    'path to config file')
flags.DEFINE_bool('live', False,
                  'Show live video or not')

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


def main(_argv):
    flags.mark_flag_as_required('config')

    record = False
    num_categories = 4
    next_active = [0] * num_categories
    record_active = 0
    freq = cv2.getTickFrequency()
    tmp_folder = None
    frame_count = 0

    args = get_webcam_arguments(FLAGS)

    min_threshold = args.get("min_threshold")
    fps = args.get("fps")
    wait_time = args.get("wait_after_message")
    min_record_time = args.get("min_record_time")
    temp_path = args.get("temp_dir")
    record_path = args.get("record_path")
    recipients = args.get("send_mail_to")
    neglect_categories = args.get("neglect_categories")
    model = tf.saved_model.load(args.get("model"))
    logging.info('...model loaded...')

    neglect = []

    if neglect_categories:
        for key, value in theft_category_index.items():
            if value["name"] in neglect_categories:
                neglect.append(value["id"])

    os.makedirs(temp_path, exist_ok=True)

    cap = VideoStream(resolution=args.get("res"), cam=args.get("camera")).start()

    width = int(cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    recover_videos(temp_path, record_path, fps, resolution=(width, height))

    mailer = SendMail(mail_id=EMAIL_ADDRESS,
                      password=EMAIL_PASSWORD,
                      recipients=recipients)

    while True:
        t1 = cv2.getTickCount()
        img = cap.read()
        image_tensor = np.expand_dims(img, axis=0)
        detections = model(image_tensor)
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()

        category = int(get_category_index(scores, classes, min_threshold=min_threshold))
        if category not in neglect and 0 < category <= num_categories:
            if time.time() > next_active[category - 1]:
                next_active[category - 1] = time.time() + wait_time
                class_name = theft_category_index[category]['name']
                mailer.send_mail(create_mail_msg(class_name), get_image_bytes(img))
            if tmp_folder is None:
                tmp_folder = os.path.join(temp_path, f"{secrets.token_hex(8)}")
                os.makedirs(tmp_folder, exist_ok=True)
                frame_count = 0
            record = True
            record_active = time.time() + min_record_time

        frame = img.copy()

        t2 = cv2.getTickCount()
        t = (t2 - t1) / freq
        cv2.putText(frame, f'FPS: {1 / t:.2f}'.format(), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0),
                    1, cv2.LINE_AA)
        local_time = time.strftime("%a %d-%b-%Y %H:%M:%S", time.localtime())
        cv2.putText(frame, f"{local_time}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255),
                    1, cv2.LINE_AA)

        if record and tmp_folder is not None:
            cv2.imwrite(
                os.path.join(tmp_folder, f"{frame_count:04}{secrets.token_hex(4)}.jpg"),
                frame)
            if time.time() > record_active:
                logging.info('...recording stopped, generating video...')
                record = False
                SaveVideo(tmp_folder, record_path, fps, (width, height)).save()
                tmp_folder = None
            frame_count += 1

        if FLAGS.live:
            frame = draw_boxes(frame, boxes, classes, scores,
                               theft_category_index, height, width,
                               min_threshold=min_threshold)
            cv2.imshow("Object Detection", cv2.resize(frame, (800, 600)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.stop()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
