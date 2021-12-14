import __load_modules  # noqa
import time
import cv2
import tensorflow as tf
import numpy as np
from src.utils import draw_boxes
from src.category import read_label_pbtxt
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string("model", None, "path to model inference graph")
flags.DEFINE_string("video", None, "path to input video")
flags.DEFINE_string("labels", None, "path to labels.txt file with detection classes")
flags.DEFINE_string("output", "data/outputs/video_output.avi", "path to output video")
flags.DEFINE_float("threshold", 0.5, "detection threshold")


def main(_argv):
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("video")
    flags.mark_flag_as_required("labels")

    labels = read_label_pbtxt(FLAGS.labels)

    start_time = time.time()
    model = tf.saved_model.load(FLAGS.model)
    end_time = time.time()
    logging.info("model loaded")
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")

    start_time = time.time()
    cap = cv2.VideoCapture(FLAGS.video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    if FLAGS.output:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        _, img = cap.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            break

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

        output_image = cv2.resize(output_image, (width, height))
        cv2.imshow("Object Detection", output_image)
        if out:
            out.write(output_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    end_time = time.time()
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
