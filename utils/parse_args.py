import json


def get_webcam_arguments(flag):
    try:
        with open(flag.config, "r") as f:
            webcam_args = json.load(f)
        return webcam_args
    except Exception as e:
        raise Exception("Error Occurred while loading config: ", e)
