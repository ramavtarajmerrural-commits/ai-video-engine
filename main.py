import os
import uuid
import asyncio
import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from edge_tts import Communicate
from groq import Groq
from moviepy.editor import AudioFileClip, ImageClip

app = FastAPI()

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

async def generate_voice(text, filename):
    communicate = Communicate(text, "hi-IN-MadhurNeural")
    await communicate.save(filename)

@app.get("/")
def home():
    return {"status": "Server is running"}

@app.get("/generate", response_class=PlainTextResponse)
def generate_video(prompt: str):
    try:
        uid = str(uuid.uuid4())[:8]

        # 1. Groq Script
        ai_resp = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Write exactly 2 simple lines in pure Hindi explaining: {prompt}. Only output the narration text."
            }],
            model="openai/gpt-oss-20b",
        )
        narration = ai_resp.choices[0].message.content.strip()

        # 2. Voice (Edge-TTS)
        audio_path = f"static/audio_{uid}.mp3"
        asyncio.run(generate_voice(narration, audio_path))

        # 3. Image (Pollinations AI)
        image_path = f"static/img_{uid}.jpg"
        img_url = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}?width=720&height=1280&nologo=true"
        res = requests.get(img_url)
        with open(image_path, "wb") as f:
            f.write(res.content)

        # 4. Video (MoviePy)
        audio_clip = AudioFileClip(audio_path)
        image_clip = ImageClip(image_path).set_duration(audio_clip.duration)
        video = image_clip.set_audio(audio_clip)

        video_path = f"static/video_{uid}.mp4"
        video.write_videofile(video_path, fps=24, codec="libx264", audio_codec="aac",ffmpeg_params=["-pix_fmt", "yuv420p"])

        return f"https://ai-video-engine-zd5t.onrender.com/{video_path}"

    except Exception as e:
        return f"Error: {str(e)}"
