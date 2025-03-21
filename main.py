import moviepy.editor as mp
from moviepy.editor import TextClip, CompositeVideoClip
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pytube

import moviepy.config as mp_config
mp_config.IMAGEMAGICK_BINARY = "/usr/bin/convert"  # Common Linux path


def get_video_and_frames(input_source, is_url=False, interval=10):
    if is_url:
        youtube = pytube.YouTube(input_source)
        stream = youtube.streams.get_lowest_resolution()
        stream.download(filename="temp_video.mp4")
        video = mp.VideoFileClip("temp_video.mp4")
    else:
        video = mp.VideoFileClip(input_source)

    # Limit to 30s-1m if longer
    if video.duration > 60:
        video = video.subclip(0, 60)  # Cap at 1 minute

    frames = []
    times = []
    for t in range(0, int(video.duration), interval):
        frame = video.get_frame(t)
        frames.append(Image.fromarray(frame))
        times.append(t)
    return video, frames, times

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_captions(images):
    captions = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

# Step 4: Build story (simulated LLM)
def build_story(captions):
    prompt = "Create a short story (200-300 words) based on these scene descriptions: " + "; ".join(captions)
    story = f"The night began when {captions[0]}. Soon, {captions[1]}. By the end, {captions[2] if len(captions) > 2 else 'everything changed'}—a tale of wonder unfolded."
    return story

# Step 5: Add captions to video
def add_captions_to_video(video, captions, times):
    clips = [video]
    for i, (caption, time) in enumerate(zip(captions, times)):
        txt_clip = TextClip(caption, fontsize=24, color='white', bg_color='black', size=(video.w, None))
        txt_clip = txt_clip.set_position(('center', 'bottom')).set_start(time).set_duration(10)  # 10s per caption
        clips.append(txt_clip)
    final_video = CompositeVideoClip(clips)
    return final_video


# Step 6: Run the tool
input_source = "aladin.mp4"
video, frames, times = get_video_and_frames(input_source, is_url=False, interval=10)
captions = generate_captions(frames)
story = build_story(captions)
final_video = add_captions_to_video(video, captions, times)

# Save the output
final_video.write_videofile("output_video.mp4", fps=24)

# Output story
print("Captions:")
for t, cap in zip(times, captions):
    print(f"{t}s: {cap}")
print("\nGenerated Story:")
print(story)

# Cleanup
import os
if os.path.exists("temp_video.mp4"):
    os.remove("temp_video.mp4")