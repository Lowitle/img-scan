import io
import cv2
import numpy as np
import imutils
from flask import Flask, request, send_file
from skimage.filters import threshold_local
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

def procesar_escaneo(img):
    # 1. Redimensionar a altura 500px para analizar contornos (lógica de image2scan)
    ratio = img.shape[0] / 500.0
    resized = imutils.resize(img, height=500)

    grayscaled = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscaled, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    grabbed = imutils.grab_contours(contours)
    sortedContours = sorted(grabbed, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for contour in sortedContours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # 2. Transformar perspectiva si detectó 4 puntos; de lo contrario, mantener imagen completa
    if screenCnt is not None:
        pts = screenCnt.reshape(4, 2) * ratio
        transformed = four_point_transform(img, pts)
    else:
        transformed = img

    # 3. Aplicar filtro adaptativo gaussiano local (scikit-image)
    transformed_grayscaled = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    T = threshold_local(transformed_grayscaled, 11, offset=10, method="gaussian")
    scanned = (transformed_grayscaled > T).astype("uint8") * 255

    return scanned

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

    pil_img = Image.fromarray(resultado)
    pdf_io = io.BytesIO()
    pil_img.save(pdf_io, format='PDF', resolution=100.0)
    pdf_io.seek(0)

    return send_file(pdf_io, mimetype='application/pdf')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
