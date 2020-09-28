import time
import os
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
from absl import app, flags, logging
from absl.flags import FLAGS
from utils.utility import load_image, draw_boxes, load_labels

flags.DEFINE_string('model', None,
                    'path to tflite model')
flags.DEFINE_string('image', None, 'path to input image')
flags.DEFINE_string('output',
                    'data/outputs/detection_output.jpg',
                    'path to output image')
flags.DEFINE_float('threshold', 0.5, 'detection threshold')


def main(_argv):
    flags.mark_flag_as_required('model')
    flags.mark_flag_as_required('image')

    interpreter = tflite.Interpreter(os.path.join(FLAGS.model, "detect.tflite"))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    h = input_details[0]['shape'][1]
    w = input_details[0]['shape'][2]

    labels = load_labels(os.path.join(FLAGS.model, "labelmap.txt"))

    image = load_image(FLAGS.image, (w, h))
    image_np = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image_np)
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    frame = image.copy()

    frame = draw_boxes(frame, boxes, classes.astype(np.int32), scores,
                       labels, w, h,
                       min_threshold=FLAGS.threshold)

    output = Image.fromarray(frame)
    output.save(FLAGS.output)
    output.show()
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
