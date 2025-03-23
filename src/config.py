"""
This module will set the configuration values
"""
import json
import dotenv


class ConfigManager:
    """
    This is a class which holds the config values, the attributes can
    be modified and accessed on run time, if allowed explicitly in
    this configuration class
    """

    def __init__(self):
        # load the .env file
        env_vars = dotenv.dotenv_values(".env")
        self._captioning_model_name = env_vars["CAPTIONING_MODEL_NAME"]
        self._slm_model_name = env_vars["SLM_MODEL_NAME"]
        self._input_vid_path = env_vars["INPUT_VIDEO_PATH"]
        self._output_path = env_vars["OUTPUT_PATH"]
        self._img_frames_dir = env_vars["IMG_FRAMES_DIR"]
        self._output_video_dir = env_vars["OUTPUT_VIDEO_DIR"]
        self._image_chunk_size = env_vars["IMAGE_CHUNK_SIZE"]
        self._status_json_dir = env_vars["STATUS_JSON_DIR"]

    @property
    def captioning_model_name(self):
        return self._captioning_model_name

    @property
    def slm_model_name(self):
        return self._slm_model_name

    @property
    def input_vid_path(self):
        return self._input_vid_path

    @property
    def output_path(self):
        return self._output_path

    @property
    def img_frames_dir(self):
        return self._img_frames_dir

    @property
    def output_video_dir(self):
        return self._output_video_dir

    @property
    def image_chunk_size(self):
        return self._image_chunk_size

    @property
    def status_json_dir(self):
        return self._status_json_dir
