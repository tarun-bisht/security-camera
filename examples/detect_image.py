import __load_modules  # noqa
import time
import tensorflow as tf
import numpy as np
import cv2
from absl import app, flags, logging
from absl.flags import FLAGS
from src.utils import draw_boxes
from src.category import read_label_pbtxt
from src.utils import load_image, preprocess_input

flags.DEFINE_string("model", None, "path to model inference graph")
flags.DEFINE_string("image", None, "path to input image")
flags.DEFINE_string("labels", None, "path to labels.txt file with detection classes")
flags.DEFINE_string(
    "output", "data/outputs/detection_output.jpg", "path to output image"
)
flags.DEFINE_float("threshold", 0.5, "detection threshold")


def main(_argv):
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("image")
    flags.mark_flag_as_required("labels")

    labels = read_label_pbtxt(FLAGS.labels)

    start_time = time.time()
    tf.keras.backend.clear_session()
    model = tf.saved_model.load(FLAGS.model)
    end_time = time.time()
    logging.info("model loaded")
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")

    image_np = load_image(FLAGS.image)
    image_tensor = np.expand_dims(image_np, axis=0)
    image_tensor = preprocess_input(image_tensor)
    height, width, _ = image_np.shape
    start_time = time.time()
    detections = model(image_tensor)
    end_time = time.time()

    boxes = detections["detection_boxes"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(np.int32)
    scores = detections["detection_scores"][0].numpy()

    output_image = draw_boxes(
        image_np.copy(),
        boxes,
        classes,
        scores,
        labels,
        height,
        width,
        min_threshold=FLAGS.threshold,
    )

    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(FLAGS.output, output_image)
    cv2.imshow("Object Detection", output_image)
    cv2.waitKey(0)
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
