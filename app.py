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

    response = client.models.generate_content(
        model='gemini-2.5-flash',
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
        texto_json = texto_json.replace("```json", "").replace("```", "").strip()

    return json.loads(texto_json)

def transformar_perspectiva(img, esquinas_norm):
    h, w = img.shape[:2]

    # Convertir coordenadas (0-1000) a píxeles exactos
    pts = np.float32([
        [esquinas_norm['top_left'][0] * w / 1000.0, esquinas_norm['top_left'][1] * h / 1000.0],
        [esquinas_norm['top_right'][0] * w / 1000.0, esquinas_norm['top_right'][1] * h / 1000.0],
        [esquinas_norm['bottom_right'][0] * w / 1000.0, esquinas_norm['bottom_right'][1] * h / 1000.0],
        [esquinas_norm['bottom_left'][0] * w / 1000.0, esquinas_norm['bottom_left'][1] * h / 1000.0]
    ])

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    ancho_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    ancho_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[0] - tl[0]) ** 2))
    max_ancho = max(int(ancho_A), int(ancho_B))

    alto_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    alto_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_alto = max(int(alto_A), int(alto_B))

    dst = np.array([
        [0, 0],
        [max_ancho - 1, 0],
        [max_ancho - 1, max_alto - 1],
        [0, max_alto - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (max_ancho, max_alto))

def aplicar_filtro_limpieza(img):
    """Blanquea sombras de la mesa preservando tintas y contraste."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    bg = cv2.GaussianBlur(l, (33, 33), 0)
    l_norm = cv2.divide(l, bg, scale=255)
    l_norm = cv2.normalize(l_norm, None, 0, 255, cv2.NORM_MINMAX)
    lab_clean = cv2.merge((l_norm, a, b))
    return cv2.cvtColor(lab_clean, cv2.COLOR_LAB2BGR)

@app.route('/escanear', methods=['POST'])
def escanear():
    if 'imagen' not in request.files:
        return "No image provided", 400

    file = request.files['imagen']
    img_bytes = file.read()

    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return "Invalid image format", 400

    try:
        # 1. Llamar a Gemini para obtener las 4 esquinas
        esquinas = obtener_esquinas_gemini(img_bytes)
        print(f"Esquinas detectadas por IA: {esquinas}", flush=True)

        # 2. Enderezar la foto en 3D
        recortado = transformar_perspectiva(img, esquinas)

        # 3. Aplicar filtro
        resultado = aplicar_filtro_limpieza(recortado)
    except Exception as e:
        print("--- ERROR EN PROCESAMIENTO ---", flush=True)
        print(traceback.format_exc(), flush=True)
        resultado = img  # Devolución de reserva si ocurre un error

    resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(resultado_rgb)

    pdf_io = io.BytesIO()
    pil_img.save(pdf_io, format='PDF', resolution=100.0)
    pdf_io.seek(0)

    return send_file(pdf_io, mimetype='application/pdf')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
