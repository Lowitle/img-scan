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

def es_cuadrilatero_valido(pts):
    """Verifica que los 4 puntos formen una hoja de papel coherente y sin distorsión extrema."""
    rect = ordenar_puntos(pts)
    (tl, tr, br, bl) = rect
    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    
    if min(w1, w2, h1, h2) == 0:
        return False
        
    aspect_ratio = max(w1, w2) / max(h1, h2)
    # Rango de proporción válido para una hoja (vertical o horizontal)
    if not (0.4 <= aspect_ratio <= 2.5):
        return False
        
    return cv2.isContourConvex(pts.astype(int))

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

def filtro_documento_color(img):
    """Aplica contraste vivo estilo escáner: blanquea fondo y oscurece tinta."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Ecualización adaptativa de luminancia (resalta texto a mano)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Curva de contraste: texto más oscuro, papel más blanco
    l_float = l_clahe.astype(np.float32)
    l_out = (l_float - 35) * (255.0 / (195.0 - 35.0))
    l_out = np.clip(l_out, 0, 255).astype(np.uint8)

    lab_final = cv2.merge((l_out, a, b))
    return cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)

def procesar_escaneo(img):
    h_orig, w_orig = img.shape[:2]
    area_total = h_orig * w_orig

    target_h = 800.0
    ratio = h_orig / target_h
    peque = cv2.resize(img, (int(w_orig / ratio), int(target_h)))

    # 1. Detectar el papel mediante espacio de color HSV (Mesa de madera vs Papel claro)
    hsv = cv2.cvtColor(peque, cv2.COLOR_BGR2HSV)
    mascara_papel = cv2.inRange(hsv, (0, 0, 90), (180, 70, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mascara_papel = cv2.morphologyEx(mascara_papel, cv2.MORPH_CLOSE, kernel)
    mascara_papel = cv2.morphologyEx(mascara_papel, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(mascara_papel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    doc_cnt = None
    for c in cnts:
        area_peque = cv2.contourArea(c)
        area_real = area_peque * (ratio ** 2)

        if area_real < (area_total * 0.25):
            continue

        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

        if len(approx) == 4 and es_cuadrilatero_valido(approx.reshape(4, 2)):
            doc_cnt = approx
            break

    # 2. Encuadre seguro sin deformación
    if doc_cnt is not None:
        pts_orig = doc_cnt.reshape(4, 2) * ratio
        recortado = transformar_perspectiva(img, pts_orig)
    else:
        # Fallback: recorte recto limpio si la geometría de esquinas falla
        if len(cnts) > 0 and (cv2.contourArea(cnts[0]) * (ratio**2)) > (area_total * 0.25):
            x, y, w, h = cv2.boundingRect(cnts[0])
            x_o, y_o = int(x * ratio), int(y * ratio)
            w_o, h_o = int(w * ratio), int(h * ratio)
            recortado = img[y_o:y_o+h_o, x_o:x_o+w_o]
        else:
            recortado = img

    # 3. Aplicar filtro de color intenso y nitidez
    return filtro_documento_color(recortado)

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

    resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(resultado_rgb)

    pdf_io = io.BytesIO()
    pil_img.save(pdf_io, format='PDF', resolution=100.0)
    pdf_io.seek(0)

    return send_file(pdf_io, mimetype='application/pdf')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
