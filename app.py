import io
import cv2
import numpy as np
from flask import Flask, request, send_file
from PIL import Image

app = Flask(__name__)

def ordenar_puntos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def transformar_perspectiva(imagen, pts):
    rect = ordenar_puntos(pts)
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
    return cv2.warpPerspective(imagen, M, (max_ancho, max_alto))

def procesar_escaneo(img):
    h_orig, w_orig = img.shape[:2]
    area_total = h_orig * w_orig
    
    # 1. Reescalar a una altura estándar para detectar contornos limpios
    target_h = 800.0
    ratio = h_orig / target_h
    new_w = int(w_orig / ratio)
    peque = cv2.resize(img, (new_w, int(target_h)))
    
    gris = cv2.cvtColor(peque, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Umbral de Otsu + Canny para separar el papel blanco del fondo de madera
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edged = cv2.Canny(blur, 50, 150)
    combinado = cv2.bitwise_or(edged, thresh)
    
    # Cerrar grietas en el borde del folio
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(combinado, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    doc_cnt = None
    for c in cnts:
        area_peque = cv2.contourArea(c)
        area_real = area_peque * (ratio ** 2)
        
        # Descartar elementos internos (debe ocupar al menos el 20% de la foto)
        if area_real < (area_total * 0.20):
            continue
            
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break

    # Si encuentra los 4 bordes del papel principal
    if doc_cnt is not None:
        pts_orig = doc_cnt.reshape(4, 2) * ratio
        recortado = transformar_perspectiva(img, pts_orig)
    else:
        recortado = img

    # Aplicar el filtro de contraste para escáner
    recortado_gris = cv2.cvtColor(recortado, cv2.COLOR_BGR2GRAY)
    escaner = cv2.adaptiveThreshold(
        recortado_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, 11
    )
    return escaner

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

    resultado = procesar_escaneo(img)

    pil_img = Image.fromarray(resultado)
    pdf_io = io.BytesIO()
    pil_img.save(pdf_io, format='PDF', resolution=100.0)
    pdf_io.seek(0)

    return send_file(pdf_io, mimetype='application/pdf')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
