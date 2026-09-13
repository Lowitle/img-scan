import os
import io
import json
import traceback
import cv2
import numpy as np
from flask import Flask, request, send_file
from PIL import Image
from google import genai
from google.genai import types

app = Flask(__name__)

def order_points(pts):
    """Ordena los 4 puntos en sentido horario: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def obtener_esquinas_gemini(img_bytes):
    """Consulta a la API de Gemini para identificar las 4 esquinas exactas del folio."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("CRÍTICO: La variable GEMINI_API_KEY no está configurada en Render.")

    client = genai.Client(api_key=api_key)
    
    prompt = (
        "Identifica las 4 esquinas exteriores del documento/folio en la imagen. "
        "Devuelve UNICAMENTE un objeto JSON con las coordenadas normalizadas (rango 0 a 1000) "
        "en este formato exacto: "
        '{"top_left": [x,y], "top_right": [x,y], "bottom_right": [x,y], "bottom_left": [x,y]}'
    )

    # Utiliza el modelo gemini-3.6-flash disponible en tu clave API
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )

    texto_json = response.text.strip()
    if texto_json.startswith("```json"):
        texto_json = texto_json.replace("
