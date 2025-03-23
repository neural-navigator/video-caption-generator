from video_processing.preprocess import get_video_and_frames
from utils.clean_up import cleanup

class VideoProcessor:
    def __init__(self,
                 conif_obj,
                 video_path,
                 captioning_obj,
                 slm_object,
                 tts_object):
        self._video_path = video_path
        self._captioning_obj = captioning_obj
        self._slm_object = slm_object
        self._tts_object = tts_object
        self._config_obj = conif_obj

    def process_video(self):
        frames_path = get_video_and_frames(video_path=self._video_path,
                             output_dir=self._config_obj.output_path,
                             interval=1)

        captions = self._captioning_obj.generate_caption(frames_path)
        del self._captioning_obj


        print(captions)
