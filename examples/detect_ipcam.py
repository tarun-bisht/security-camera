import __load_modules  # noqa
import time
import cv2
import tensorflow as tf
import numpy as np
from src.category import read_label_pbtxt
from src.utils import draw_boxes
from absl import app, flags, logging
from absl.flags import FLAGS
from src.utility import VideoStream

flags.DEFINE_string("model", None, "path to model inference graph")
flags.DEFINE_string("output", "data/outputs/ipcam_output.avi", "path to output video")
flags.DEFINE_string("ip", None, "IP address of camera")
flags.DEFINE_integer("port", 8080, "Port in which camera is running")
flags.DEFINE_string("username", None, "Username to access camera stream")
flags.DEFINE_string("password", None, "Password to access camera stream")


def main(_argv):
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("ip")
    flags.mark_flag_as_required("labels")

    labels = read_label_pbtxt(FLAGS.labels)

    stream_url = f"rtsp://{FLAGS.ip}:{FLAGS.port}/h264_ulaw.sdp"
    if FLAGS.username and FLAGS.password:
        stream_url = f"rtsp://{FLAGS.username}:{FLAGS.password}@{FLAGS.ip}:{FLAGS.port}/h264_ulaw.sdp"

    start_time = time.time()
    model = tf.saved_model.load(FLAGS.model)
    end_time = time.time()
    logging.info("model loaded")
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")

    start_time = time.time()
    cap = VideoStream(cam=stream_url).start()

    width = int(cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    if FLAGS.output:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        img = cap.read()
        image_tensor = np.expand_dims(img, axis=0)
        detections = model(image_tensor)

        boxes = detections["detection_boxes"][0].numpy()
        classes = detections["detection_classes"][0].numpy().astype(np.int32)
        scores = detections["detection_scores"][0].numpy()

        output_image = draw_boxes(
            img.copy(),
            boxes,
            classes,
            scores,
            labels,
            height,
            width,
            min_threshold=FLAGS.threshold,
        )
        cv2.imshow("Object Detection", cv2.resize(output_image, (800, 600)))
        if out:
            out.write(output_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.stop()
    end_time = time.time()
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
