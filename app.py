from flask import Flask, request, jsonify
from PIL import Image
import torch
from ultralytics import YOLO
from io import BytesIO
import requests
import os
import time

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
    # Verificar si se ha enviado un archivo o una URL
    if 'url' not in request.json:
        return jsonify({'error': 'No URL provided'}), 400
    image_url = request.json['url']
    response = requests.get(image_url)
    
    if response.status_code != 200:
        return jsonify({'error':'Could not retrieve image'}),400
    img = Image.open(BytesIO(response.content)).convert('RGB')
    start_time = time.time()
    # Realizar la predicción en la imagen
    results = model(img)
    end_time = time.time()
    print(f"Tiempo de procesamiento: {end_time - start_time} segundos")

# Medir el uso de memoria (opcional, depende de tu entorno)
# Puedes usar la biblioteca psutil para obtener información sobre el uso de memoria
    import psutil
    print(f"Uso de memoria: {psutil.virtual_memory().percent}%")
    
    # Procesar los resultados de la detección
    detections = []
    for pred in results[0].boxes:
        x1, y1, x2, y2 = [int(coord) for coord in pred.xyxy.cpu().numpy().astype(int)[0]]
        label_index = int(pred.cls.item())
        confidence = float(pred.conf.item())
        
        # Verificar si el índice está dentro de las etiquetas personalizadas
        if label_index < len(custom_labels):
            label_name = custom_labels[label_index]
        else:
            label_name = f'Unknown({label_index})'
        detections.append({
            'label': label_name,
            'confidence': confidence,
            'box': [x1, y1, x2, y2]
        })
    
    # Devolver los resultados como JSON
    return jsonify(detections)
    # Verificar si se ha enviado un archivo
   

if __name__ == "__main__":
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
