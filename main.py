import os
import io
import uuid
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv

# --- 1. LOAD SECURE KEYS ---
load_dotenv()

# Reverted to your original .env variable names
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") 

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ ERROR: Supabase keys not found. Check your .env or Render Environment Variables!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. INITIALIZE APP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. LOAD TFLITE MODEL ---
interpreter = tflite.Interpreter(model_path="flower_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # --- 4. AI INFERENCE ---
    contents = await file.read()
    
    img = Image.open(io.BytesIO(contents)).convert('RGB').resize((180, 180))
    
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) 

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Softmax logic using NumPy
    exp_preds = np.exp(predictions[0] - np.max(predictions[0]))
    score = exp_preds / exp_preds.sum()
    
    species = CLASS_NAMES[np.argmax(score)]
    conf = round(float(np.max(score)) * 100, 2)

    # --- 5. STORAGE ---
    file_id = str(uuid.uuid4())
    file_path = f"{file_id}.jpg"
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    
    supabase.storage.from_("flower-images").upload(
        path=file_path,
        file=img_byte_arr.getvalue(),
        file_options={"content-type": "image/jpeg"}
    )
    
    image_url = supabase.storage.from_("flower-images").get_public_url(file_path)

    # --- 6. DATABASE LOGGING ---
    data = {
        "species": species,
        "confidence": conf,
        "image_url": image_url
    }
    supabase.table("flower_scans").insert(data).execute()

    return {
        "species": species, 
        "confidence": conf, 
        "image_url": image_url
    }