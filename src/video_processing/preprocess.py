import os
import moviepy.editor as mp
from moviepy.editor import TextClip, CompositeVideoClip
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pytube

import moviepy.config as mp_config
mp_config.IMAGEMAGICK_BINARY = "/usr/bin/convert"

# TODO: Document the movie py configuration

def download_youtube_video(url):
    raise NotImplementedError


def get_video_and_frames(video_path, output_dir, interval=1):
    video = mp.VideoFileClip(video_path)
    video_path = os.path.join(output_dir, video_path.split("/")[-1])
    os.makedirs(video_path, exist_ok=True)

    for t in range(0, int(video.duration), interval):
        frame = Image.fromarray(video.get_frame(t))
        frame.save(os.path.join(video_path, f"{t}.png"))

    return video_path


if __name__ == "__main__":
    # get_video_and_frames("input_data/sample-vid.mp4")
    pass