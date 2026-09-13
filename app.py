from flask import Flask, request, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

@app.route('/escanear', methods=['POST'])
def escanear():
    # 1. Recibir la imagen desde Google Apps Script
    file = request.files['imagen']
    img_bytes = file.read()
    
    # 2. Convertir bytes a imagen para OpenCV
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # --- AQUÍ VA LA MAGIA DE OPENCV ---
    # Convertir a escala de grises
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Filtro para que parezca escáner (Adaptive Thresholding)
    escaner = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 21, 10)
    # -----------------------------------
    
    # 3. Volver a convertir a imagen JPG para enviarla de vuelta
    _, buffer = cv2.imencode('.jpg', escaner)
    io_buf = io.BytesIO(buffer)
    
    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)