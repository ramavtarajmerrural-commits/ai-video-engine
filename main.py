import os
import json
import asyncio
import urllib.parse
import subprocess
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq
import edge_tts
import httpx

app = FastAPI(title="Flow Pro AI Video Engine")

os.makedirs("generated_videos", exist_ok=True)
os.makedirs("temp_assets", exist_ok=True)
app.mount("/videos", StaticFiles(directory="generated_videos"), name="videos")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

VOICE_MAP = {
    "hi": "hi-IN-MadhurNeural",
    "en": "en-US-ChristopherNeural",
    "bn": "bn-IN-BashkarNeural",
    "te": "te-IN-MohanNeural",
    "mr": "mr-IN-ManoharNeural",
    "ta": "ta-IN-ValluvarNeural",
    "gu": "gu-IN-NiranjanNeural",
    "ur": "ur-IN-SalmanNeural"
}

class VideoRequest(BaseModel):
    prompt: str
    language: str = "hi"
    aspect_ratio: str = "9:16"

@app.post("/generate-video")
async def generate_video(req: VideoRequest, request: Request):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key nahi mili")
    
    clean_prompt = req.prompt.strip()[:300]
    target_lang = req.language.lower()
    voice_name = VOICE_MAP.get(target_lang, "hi-IN-MadhurNeural")
    
    system_prompt = f"""
    You are a cinematic video director. Break the topic into exactly 3 sequential scenes.
    Write the narration strictly in language code: '{target_lang}' (e.g. Hindi in Devanagari).
    Image prompts must always be in English, highly detailed, photorealistic, 8k, cinematic lighting.
    Return ONLY a valid JSON object matching this schema:
    {{
      "scenes": [
        {{"scene_id": 1, "image_prompt": "Cinematic visual description in English", "narration": "Script line in {target_lang}"}},
        {{"scene_id": 2, "image_prompt": "Cinematic visual description in English", "narration": "Script line in {target_lang}"}},
        {{"scene_id": 3, "image_prompt": "Cinematic visual description in English", "narration": "Script line in {target_lang}"}}
      ]
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Topic: {clean_prompt}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        storyboard = json.loads(completion.choices[0].message.content)
        scenes = storyboard.get("scenes", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storyboard error: {str(e)}")

    task_id = f"vid_{os.urandom(4).hex()}"
    width, height = (720, 1280) if req.aspect_ratio == "9:16" else (1280, 720)
    scene_clips = []
    
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        for idx, sc in enumerate(scenes):
            img_path = f"temp_assets/{task_id}_img_{idx}.jpg"
            audio_path = f"temp_assets/{task_id}_aud_{idx}.mp3"
            clip_path = f"temp_assets/{task_id}_clip_{idx}.mp4"
            
            encoded_prompt = urllib.parse.quote(sc["image_prompt"])
            img_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&model=flux&nologo=true&seed={os.urandom(2).hex()}"
            
            img_res = await http_client.get(img_url)
            with open(img_path, "wb") as f:
                f.write(img_res.content)
                
            tts = edge_tts.Communicate(sc["narration"], voice_name)
            await tts.save(audio_path)
            
            probe_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {audio_path}"
            duration = float(subprocess.check_output(probe_cmd, shell=True).decode().strip())
            duration = max(duration, 3.0)
            
            zoom_filter = f"scale=8000:-1,zoompan=z='min(zoom+0.0015,1.25)':d={int(duration*25)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height},setsar=1"
            clip_cmd = (
                f"ffmpeg -y -loop 1 -t {duration} -i {img_path} -i {audio_path} "
                f"-vf \"{zoom_filter}\" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest {clip_path}"
            )
            subprocess.run(clip_cmd, shell=True, check=True)
            scene_clips.append(clip_path)

    concat_list = f"temp_assets/{task_id}_list.txt"
    with open(concat_list, "w") as f:
        for clip in scene_clips:
            f.write(f"file '{os.path.abspath(clip)}'\n")
            
    final_filename = f"{task_id}_final.mp4"
    final_output = f"generated_videos/{final_filename}"
    
    concat_cmd = f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c copy {final_output}"
    subprocess.run(concat_cmd, shell=True, check=True)
    
    for clip in scene_clips:
        try:
            os.remove(clip)
        except:
            pass
    try:
        os.remove(concat_list)
        for idx in range(len(scenes)):
            os.remove(f"temp_assets/{task_id}_img_{idx}.jpg")
            os.remove(f"temp_assets/{task_id}_aud_{idx}.mp3")
    except:
        pass

    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/videos/{final_filename}"
