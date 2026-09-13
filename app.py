import os
import io
import json
import cv2
import numpy as np
from flask import Flask, request, send_file
from PIL import Image
from google import genai
from google.genai import types

app = Flask(__name__)

# Inicializar cliente de la API con tu clave
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def obtener_esquinas_ia(img_bytes):
    """
    Solicita a la IA que detecte las 4 esquinas del documento en la foto.
    """
    prompt = (
        "Analiza la imagen y detecta el papel principal (documento/albarán). "
        "Devuelve UNICAMENTE un objeto JSON con las coordenadas normalizadas (de 0 a 1000) "
        "de las 4 esquinas del papel en este orden: "
        '{"top_left": [x,y], "top_right": [x,y], "bottom_right": [x,y], "bottom_left": [x,y]}'
    )

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    
    # Parsear respuesta JSON de la IA
    datos = json.loads(response.text)
    return datos

def transformar_perspectiva_ia(img, esquinas_norm):
    h, w = img.shape[:2]
    
    # Convertir coordenadas normalizadas (0-1000) a píxeles reales de la foto
    tl = [esquinas_norm['top_left'][0] * w / 1000.0, esquinas_norm['top_left'][1] * h / 1000.0]
    tr = [esquinas_norm['top_right'][0] * w / 1000.0, esquinas_norm['top_right'][1] * h / 1000.0]
    br = [esquinas_norm['bottom_right'][0] * w / 1000.0, esquinas_norm['bottom_right'][1] * h / 1000.0]
    bl = [esquinas_norm['bottom_left'][0] * w / 1000.0, esquinas_norm['bottom_left'][1] * h / 1000.0]

    pts1 = np.float32([tl, tr, br, bl])

    # Calcular dimensiones finales del papel enderezado
    ancho_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    ancho_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[0] - tl[0]) ** 2))
    max_ancho = max(int(ancho_A), int(ancho_B))

    alto_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    alto_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_alto = max(int(alto_A), int(alto_B))

    pts2 = np.float32([[0, 0], [max_ancho - 1, 0], [max_ancho - 1, max_alto - 1], [0, max_alto - 1]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (max_ancho, max_alto))

@app.route('/escanear', methods=['POST'])
def escanear():
    if 'imagen' not in request.files:
        return "No image", 400

    file = request.files['imagen']
    img_bytes = file.read()

    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return "Invalid image", 400

    try:
        # 1. Obtener esquinas exactas mediante la IA
        esquinas = obtener_esquinas_ia(img_bytes)
        
        # 2. Recortar y enderezar en 3D
        recortado = transformar_perspectiva_ia(img, esquinas)
    except Exception as e:
        # Si la llamada a la API falla por red, usa la foto sin recortar como respaldo
        recortado = img

    # 3. Convertir a PDF limpia
    resultado_rgb = cv2.cvtColor(recortado, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(resultado_rgb)

    pdf_io = io.BytesIO()
    pil_img.save(pdf_io, format='PDF', resolution=100.0)
    pdf_io.seek(0)

    return send_file(pdf_io, mimetype='application/pdf')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
