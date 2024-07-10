from flask import Flask, request, jsonify
from PIL import Image
import torch
from ultralytics import YOLO
from io import BytesIO
import os

app = Flask(__name__)

# Cargar el modelo YOLOv5
model = YOLO('model/best.pt')

# Definir etiquetas personalizadas si es necesario
custom_labels = ['Cervicabra', 'Chupil', 'Condor', 'GallaretaAndina', 'GanadoVacuno', 'Humano', 'OsoAnteojos', 'PatoZambullidorGrande', 'Puma', 'TucanAndinoPiquilaminado', 'VenadoColaBlanca', 'ZorrilloEspaldaBlancaSureño', 'ZorroCulpeo']

@app.route('/', methods=['GET'])
def saludar():
    return jsonify({'mensaje': '¡Hola desde mi aplicación en Render!'})

@app.route('/detect', methods=['POST'])
def detect_objects():
    # Verificar si se ha enviado un archivo
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Leer la imagen desde la solicitud
    image_file = request.files['image']
    img = Image.open(BytesIO(image_file.read())).convert('RGB')

    # Realizar la predicción en la imagen
    results = model(img)

    # Procesar los resultados de la detección
    detections = []
    for pred in results[0].boxes:
        x1, y1, x2, y2 = pred.xyxy.cpu().numpy().astype(int)[0]
        label_index = int(pred.cls.item())
        confidence = float(pred.conf.item())
        # Verificar si el índice está dentro de las etiquetas personalizadas
        if label_index < len(custom_labels):
            label_name = custom_labels[label_index]
        else:
            label_name = f'Unknown({label_index})'
        
        # Asegurarse de que x1, y1, x2, y2 son enteros
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        detections.append({
            'label': label_name,
            'confidence': float(confidence),
            'box': [x1, y1, x2, y2]
        })
    
    # Devolver los resultados como JSON
    return jsonify(detections)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
    #port = int(os.environ.get('PORT', 5000))
    #app.run(host='0.0.0.0', port=port, debug=True)
