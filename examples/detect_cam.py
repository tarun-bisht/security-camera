import __load_modules  # noqa
import time
import cv2
import tensorflow as tf
import numpy as np
from utils.utility import draw_boxes
from utils.category import theft_category_index
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string("model", None, "path to model inference graph")
flags.DEFINE_string("output", "data/outputs/cam_output.avi", "path to output video")
flags.DEFINE_integer("cam", 0, "camera number or id to access")
flags.DEFINE_float("threshold", 0.5, "detection threshold")


def main(_argv):
    flags.mark_flag_as_required("model")

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    start_time = time.time()
    model = tf.saved_model.load(FLAGS.model)
    end_time = time.time()
    logging.info("model loaded")
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")

    start_time = time.time()
    cap = cv2.VideoCapture(FLAGS.cam)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    if FLAGS.output:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        _, img = cap.read()
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
            theft_category_index,
            height,
            width,
            min_threshold=FLAGS.threshold,
        )

        cv2.imshow("Object Detection", cv2.resize(output_image, (800, 600)))
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
