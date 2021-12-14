import json
from absl.flags import FLAGS
from src.category import read_label_pbtxt


def get_security_cam_arguments(flag: FLAGS) -> tuple:
    """parse arguments from config file

    Args:
        flag (FLAGS): command line arguments

    Raises:
        Exception: Error when config file cannot be loaded

    Returns:
        tuple: (config file args in form of dictionary, labels as dictionary)
    """
    try:
        with open(flag.config, "r") as f:
            cfg = json.load(f)
        args = {}
        cam_args = cfg["camera"]
        if cam_args["network"].get("ip") is not None:
            stream_url = cam_args["network"]["url_format"]
            stream_url = stream_url.replace("<ip>", cam_args["network"]["ip"])
            stream_url = stream_url.replace("<port>", cam_args["network"]["port"])
            username = cam_args["network"]["username"]
            password = cam_args["network"]["password"]
            if (username and password) or (username != "" and password != ""):
                stream_url = stream_url.replace("<username>", username)
                stream_url = stream_url.replace("<password>", password)
            else:
                stream_url = stream_url.replace("<username>:<password>@", "")
            args["cam"] = stream_url
        else:
            args["cam"] = cam_args["id"]
        args["model"] = cfg["detector"]["detector_model_path"]
        args["fps"] = cfg["detector"]["fps"]
        args["res"] = cfg["detector"]["resolution"]
        args["recording_save_path"] = cfg["detector"]["recording_save_path"]
        args["temp_dir"] = cfg["detector"]["temp_dir"]
        args["neglect_categories"] = cfg["detector"]["neglect_categories"]
        args["min_threshold"] = cfg["detector"]["min_threshold"]
        args["min_recording_time"] = cfg["detector"]["min_recording_time"]
        args["wait_after_message_send"] = cfg["notifier"]["wait_after_message_send"]
        args["send_mail_to"] = cfg["notifier"]["send_mail_to"]
        labels = read_label_pbtxt(cfg["detector"]["labels_path"])
        return args, labels
    except Exception as e:
        raise Exception("An error occurred while loading config: ", e)
