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

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def order_points(pts):
    """Ordena los 4 puntos exactamente: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def obtener_datos_ia(img_bytes):
    """Solicita a Gemini el área del documento y sus esquinas aproximadas."""
    prompt = (
        "Detect the main paper document in the image. "
        "Return ONLY a JSON object with: "
        '"bbox": {"ymin": int, "xmin": int, "ymax": int, "xmax": int} (range 0 to 1000) and '
        '"corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (range 0 to 1000).'
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
    return json.loads(response.text)

def obtener_esquinas_exactas(img, datos_ia):
    h, w = img.shape[:2]
    
    # 1. Delimitar la búsqueda usando el área que indicó la IA (con un margen del 5%)
    bbox = datos_ia['bbox']
    pad = 30
    ymin = max(0, int(bbox['ymin'] * h / 1000.0) - pad)
    xmin = max(0, int(bbox['xmin'] * w / 1000.0) - pad)
    ymax = min(h, int(bbox['ymax'] * h / 1000.0) + pad)
    xmax = min(w, int(bbox['xmax'] * w / 1000.0) + pad)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[ymin:ymax, xmin:xmax] = 255

    # 2. OpenCV busca el contorno exacto SOLO dentro de la zona delimitada por la IA
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 150)
    edged_masked = cv2.bitwise_and(edged, edged, mask=mask)

    cnts, _ = cv2.findContours(edged_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            # Encontró las 4 esquinas exactas con OpenCV en el área restringida
            return order_points(approx.reshape(4, 2))

    # Fallback: Si el papel está roto o arrugado, usa las esquinas ordenadas de la IA
    corners_ia = np.array(datos_ia['corners'], dtype="float32")
    corners_ia[:, 0] = corners_ia[:, 0] * w / 1000.0
    corners_ia[:, 1] = corners_ia[:, 1] * h / 1000.0
    return order_points(corners_ia)

def transformar_perspectiva(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[0] - tl[0]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

def filtro_escaner_limpio(img):
    """Limpia el fondo dejándolo blanco estilo Google Drive sin quemar la tinta."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (35, 35), 0)
    norm = cv2.divide(gray, bg, scale=255)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

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
        # 1. IA + OpenCV encuentran las 4 esquinas perfectas
        datos_ia = obtener_datos_ia(img_bytes)
        esquinas = obtener_esquinas_exactas(img, datos_ia)
        
        # 2. Enderezar en 3D
        recortado = transformar_perspectiva(img, esquinas)
        
        # 3. Aplicar filtro escáner
        resultado = filtro_escaner_limpio(recortado)
    except Exception as e:
        resultado = img

    resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(resultado_rgb)

    pdf_io = io.BytesIO()
    pil_img.save(pdf_io, format='PDF', resolution=100.0)
    pdf_io.seek(0)

    return send_file(pdf_io, mimetype='application/pdf')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
