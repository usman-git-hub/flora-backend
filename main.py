import os
import io
import uuid
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv

# --- 1. LOAD SECURE KEYS ---
# This looks for a file named ".env" in your folder
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ ERROR: Supabase keys not found. Check your .env file!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. INITIALIZE APP ---
app = FastAPI()

# Allows your Next.js frontend to talk to this Python backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your AI model
MODEL = tf.keras.models.load_model('my_optimized_flower_model.keras')
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # --- 3. AI INFERENCE ---
    contents = await file.read()
    # Prepare image for the model (RGB, 180x180)
    img = Image.open(io.BytesIO(contents)).convert('RGB').resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = MODEL.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    species = CLASS_NAMES[np.argmax(score)]
    conf = round(float(np.max(score)) * 100, 2)

    # --- 4. CONSTANT LEARNING (Storage) ---
    # Create a unique name for the image
    file_id = str(uuid.uuid4())
    file_path = f"{file_id}.jpg"
    
    # Convert image back to bytes to send to Supabase
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    
    # Upload to "flower-images" bucket
    supabase.storage.from_("flower-images").upload(
        path=file_path,
        file=img_byte_arr.getvalue(),
        file_options={"content-type": "image/jpeg"}
    )
    
    # Get the public link for the image
    image_url = supabase.storage.from_("flower-images").get_public_url(file_path)

    # --- 5. DATABASE LOGGING ---
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