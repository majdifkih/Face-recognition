from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder

@app.route('/analyze', methods=['POST'])
def analyze_face():
    try:
        # Récupérer le fichier
        if 'img' in request.files:
            file = request.files['img']
            file.save('temp_image.jpg')
            img_path = 'temp_image.jpg'
        else:
            data = request.json
            img_path = data['img_path']
        
        # Analyser
        result = DeepFace.analyze(img_path, actions=['age', 'gender', 'emotion', 'race'])
        
        # Convertir les numpy types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        result = convert_numpy(result)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)