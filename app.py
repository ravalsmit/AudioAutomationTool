import os
import logging
from flask import Flask, request, render_template, send_from_directory
import librosa
import numpy as np
from PIL import Image, ImageEnhance
from moviepy.editor import ImageSequenceClip
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)

def adjust_image(image, brightness_factor, exposure_factor):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(exposure_factor)
    return np.array(image)

def detect_beats(audio_path):
    y, sr = librosa.load(audio_path)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return beats / sr

def create_video(image_path, audio_path, output_video_path):
    try:
        logging.info("Starting video creation")
        beats = detect_beats(audio_path)
        image = Image.open(image_path).convert('RGB')
        
        # Resize image to speed up processing
        image = image.resize((image.width // 2, image.height // 2))

        duration = librosa.get_duration(filename=audio_path)
        fps = 24
        total_frames = int(duration * fps)
        logging.info(f"Total frames: {total_frames}, Duration: {duration} seconds")

        def process_frame(frame_num):
            time = frame_num / fps
            brightness_factor = 1 + 0.5 * np.sin(2 * np.pi * beats * time).sum()
            exposure_factor = 1 + 0.5 * np.cos(2 * np.pi * beats * time).sum()
            frame = adjust_image(image, brightness_factor, exposure_factor)
            return frame

        with ThreadPoolExecutor() as executor:
            frames = list(executor.map(process_frame, range(total_frames)))

        logging.info("Frames created, starting video compilation")
        clip = ImageSequenceClip([Image.fromarray(frame) for frame in frames], fps=fps)
        clip.write_videofile(output_video_path, audio=audio_path)
        logging.info("Video created successfully")
    except Exception as e:
        logging.error(f"Error during video creation: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files or 'audio' not in request.files:
        return 'No file part'
    image = request.files['image']
    audio = request.files['audio']
    if image.filename == '' or audio.filename == '':
        return 'No selected file'
    if image and audio:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
        image.save(image_path)
        audio.save(audio_path)
        output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_video.mp4')
        create_video(image_path, audio_path, output_video_path)
        if os.path.exists(output_video_path):
            return send_from_directory(app.config['OUTPUT_FOLDER'], 'output_video.mp4')
        else:
            return 'Video creation failed'

if __name__ == '__main__':
    app.run(debug=True)
