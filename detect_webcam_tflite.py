import time
import cv2
import tensorflow as tf
import numpy as np
from utils.utility import draw_boxes
from utils.category import theft_category_index
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('model', None,
                    'path to tflite model')
flags.DEFINE_integer('cam', 0, 'camera number to access')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_float('threshold', 0.5, 'detection threshold')


def main(_argv):
    flags.mark_flag_as_required('model')

    start_time = time.time()
    interpreter = tf.lite.Interpreter(FLAGS.model)
    end_time = time.time()
    logging.info('model loaded')
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    h = input_details[0]['shape'][1]
    w = input_details[0]['shape'][2]

    interpreter.resize_tensor_input(input_details[0]['index'],
                                    [1, 320, 320, 3])
    interpreter.allocate_tensors()

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
        interpreter.set_tensor(input_details[0]['index'], image_tensor)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[3]['index'])[0]
        classes = interpreter.get_tensor(output_details[4]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]

        frame = img.copy()

        frame = draw_boxes(frame, boxes, classes.astype(np.int32), scores,
                           theft_category_index, w, h,
                           min_threshold=FLAGS.threshold)

        frame = cv2.resize(frame, (width, height))
        cv2.imshow("Object Detection", frame)
        if out:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    end_time = time.time()
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
