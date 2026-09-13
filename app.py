import io
import cv2
import numpy as np
import imutils
from flask import Flask, request, send_file
from PIL import Image

app = Flask(__name__)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
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
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def aplicar_filtro_escaner_color(img):
    """
    Blanquea el papel y elimina sombras sin pixelar el texto a mano
    y manteniendo los colores del documento (logos y sellos).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplanar iluminación de fondo
    bg = cv2.GaussianBlur(l, (45, 45), 0)
    l_clean = cv2.divide(l, bg, scale=255)
    l_clean = cv2.normalize(l_clean, None, 0, 255, cv2.NORM_MINMAX)
    
    # Recombinar canales preservando color
    lab_clean = cv2.merge((l_clean, a, b))
    res_bgr = cv2.cvtColor(lab_clean, cv2.COLOR_LAB2BGR)
    
    # Suave enfoque de nitidez para el bolígrafo
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(res_bgr, -1, kernel_sharp)

def procesar_escaneo(img):
    h_orig, w_orig = img.shape[:2]
    area_total_orig = h_orig * w_orig
    
    # Redimensionar a altura 600px para análisis ligero
    ratio = h_orig / 600.0
    resized = imutils.resize(img, height=600)
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 120)
    
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    screenCnt = None
    
    for c in cnts:
        area_peque = cv2.contourArea(c)
        area_real = area_peque * (ratio ** 2)
        
        # FILTRO DE SEGURIDAD: El recuadro debe ocupar al menos el 35% de la foto.
        # Esto impide automáticamente recortar el logo "CF" o cajas pequeñas.
        if area_real < (area_total_orig * 0.35):
            continue
            
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            screenCnt = approx
            break
            
    # Si encuentra los 4 puntos del papel completo
    if screenCnt is not None:
        pts = screenCnt.reshape(4, 2) * ratio
        recortado = four_point_transform(img, pts)
    else:
        # Fallback de seguridad: si no detecta las 4 esquinas del folio,
        # recorta el cuadro exterior principal o mantiene la foto sin deformar
        recortado = img
        for c in cnts:
            area_real = cv2.contourArea(c) * (ratio ** 2)
            if area_real >= (area_total_orig * 0.35):
                x, y, w, h = cv2.boundingRect(c)
                x_o, y_o = int(x * ratio), int(y * ratio)
                w_o, h_o = int(w * ratio), int(h * ratio)
                recortado = img[y_o:y_o+h_o, x_o:x_o+w_o]
                break
                
    return aplicar_filtro_escaner_color(recortado)

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

    resultado = procesar_escaneo(img)

    resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(resultado_rgb)
    
    pdf_io = io.BytesIO()
    pil_img.save(pdf_io, format='PDF', resolution=100.0)
    pdf_io.seek(0)

    return send_file(pdf_io, mimetype='application/pdf')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
