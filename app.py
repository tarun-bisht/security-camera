import os
import time
import cv2
import secrets
import tensorflow as tf
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from src.parse_args import get_security_cam_arguments
from src.utils import (
    VideoStream,
    draw_boxes,
    create_mail_msg,
    recover_videos,
    get_category_index,
    SaveVideo,
    SendMail,
    get_image_bytes,
)

flags.DEFINE_string(
    "config", "configs/security_cam.cfg", "path to config file for application"
)
flags.DEFINE_bool("live", True, "Show live feed or not")

# Read email and password to send mail to recepients from system environment so create these fields as system environment variables
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


def main(_argv):
    # helper variables
    record = False
    num_categories = 4
    next_active = [0] * num_categories
    record_active = 0
    freq = cv2.getTickFrequency()
    tmp_folder = None
    frame_count = 0

    # parse config file arguments
    args, security_cam_category_index = get_security_cam_arguments(FLAGS)
    min_threshold = args.get("min_threshold")
    fps = args.get("fps")
    wait_time = args.get("wait_after_message_send")
    min_record_time = args.get("min_recording_time")
    temp_path = args.get("temp_dir")
    record_path = args.get("recording_save_path")
    recipients = args.get("send_mail_to")
    neglect_categories = args.get("neglect_categories")
    tf.keras.backend.clear_session()
    model = tf.saved_model.load(args.get("model"))
    logging.info("...model loaded...")

    # collect index of categories to neglect (do not detect) from all categories in labels model trained
    neglect = []
    if neglect_categories:
        for key, value in security_cam_category_index.items():
            if value["name"] in neglect_categories:
                neglect.append(value["id"])

    # create a temp directory to save frames when intruder detected to create video.
    os.makedirs(temp_path, exist_ok=True)

    # initiating stream capture
    cap = VideoStream(resolution=args.get("res"), cam=args.get("cam")).start()

    width = int(cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # recover previously failed videos whose frames are currently stored in temp folder. Video generation failed at that time because of some system failure
    recover_videos(temp_path, record_path, fps, resolution=(width, height))

    # initiating mailer
    mailer = SendMail(
        mail_id=EMAIL_ADDRESS, password=EMAIL_PASSWORD, recipients=recipients
    )

    # detect intruder in stream
    while True:
        t1 = cv2.getTickCount()
        img = cap.read()
        image_tensor = np.expand_dims(img, axis=0)
        detections = model(image_tensor)
        boxes = detections["detection_boxes"][0].numpy()
        classes = detections["detection_classes"][0].numpy().astype(np.int32)
        scores = detections["detection_scores"][0].numpy()

        # draw bounding box in frame
        frame = img.copy()
        frame = draw_boxes(
            frame,
            boxes,
            classes,
            scores,
            security_cam_category_index,
            height,
            width,
            min_threshold=min_threshold,
            put_label=False,
        )

        # when a new category detected start recording video and send mail to notify and record till category is being detected.
        # sends a mail for every new category.
        category = int(get_category_index(scores, classes, min_threshold=min_threshold))
        if category not in neglect and 0 < category <= num_categories:
            if time.time() > next_active[category - 1]:
                next_active[category - 1] = time.time() + wait_time
                class_name = security_cam_category_index[category]["name"]
                mailer.send_mail(create_mail_msg(class_name), get_image_bytes(frame))
            if tmp_folder is None:
                tmp_folder = os.path.join(temp_path, f"{secrets.token_hex(8)}")
                os.makedirs(tmp_folder, exist_ok=True)
                frame_count = 0
            record = True
            record_active = time.time() + min_record_time

        # print local time to frame
        local_time = time.strftime("%a %d-%b-%Y %H:%M:%S", time.localtime())
        cv2.putText(
            frame,
            f"{local_time}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        # calculate fps and print it to frame
        t2 = cv2.getTickCount()
        t = (t2 - t1) / freq
        cv2.putText(
            frame,
            f"FPS: {1 / t:.2f}".format(),
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        # saving recording frames
        if record and tmp_folder is not None:
            cv2.imwrite(
                os.path.join(
                    tmp_folder, f"{frame_count:08}_{secrets.token_hex(4)}.jpg"
                ),
                frame,
            )
            if time.time() > record_active:
                logging.info("...recording stopped, generating video...")
                record = False
                # create video from frames in a new thread
                SaveVideo(tmp_folder, record_path, fps, (width, height)).save()
                tmp_folder = None
            frame_count += 1

        # if live enable then show UI
        if FLAGS.live:
            cv2.imshow("Intelligent Security Camera", cv2.resize(frame, (800, 600)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.stop()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        logging.error("Exiting")
