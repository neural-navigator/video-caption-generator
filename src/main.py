from config import ConfigManager

from contextlib import asynccontextmanager
from fastapi import FastAPI

from config import ConfigManager
from huggingface_hub.file_download import get_session

from story_generator.story_writer import StoryGenerator

session = get_session()
session.verify = False


from service import VideoProcessor
from caption_generator.caption_gen import CaptionGenerator

config = ConfigManager()
cp = CaptionGenerator(config.captioning_model_name)
slm=0
# slm = StoryGenerator(config.slm_model_name)
tts = 0

vp = VideoProcessor(config,"input_data/sample-vid.mp4", cp, slm, tts)
vp.process_video()

