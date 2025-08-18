from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import json
import os
import tempfile

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
        img_path = None
        temp_file = None
        
        # Gestion des différents types de requêtes
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Fichier uploadé via form-data
            if 'img' in request.files:
                file = request.files['img']
                if file.filename == '':
                    return jsonify({"error": "Aucun fichier sélectionné"}), 400
                
                # Créer un fichier temporaire
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                file.save(temp_file.name)
                temp_file.close()  # fermer avant analyse
                img_path = temp_file.name
            else:
                return jsonify({"error": "Aucun fichier 'img' trouvé"}), 400
                
        elif request.content_type == 'application/json':
            # Données JSON
            data = request.json
            if 'img_path' in data:
                img_path = data['img_path']
            else:
                return jsonify({"error": "Champ 'img_path' manquant"}), 400
        else:
            return jsonify({"error": "Type de contenu non supporté. Utilisez multipart/form-data ou application/json"}), 400
        
        # Vérifier que le fichier existe
        if not os.path.exists(img_path):
            return jsonify({"error": f"Fichier non trouvé: {img_path}"}), 400
        
        # Analyser avec DeepFace
        result = DeepFace.analyze(img_path, actions=['age', 'gender', 'emotion', 'race'])
        
        # Convertir les types numpy
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
        
        # Nettoyer le fichier temporaire
        if temp_file:
            os.unlink(temp_file.name)
        
        return jsonify(result)
        
    except Exception as e:
        # Nettoyer en cas d'erreur
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        return jsonify({"error": str(e)}), 500

@app.route('/verify', methods=['POST'])
def verify_faces():
    temp_files = []
    try:
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Vérifier la présence des fichiers
            if 'img1' not in request.files or 'img2' not in request.files:
                return jsonify({"error": "Fichiers 'img1' et 'img2' requis"}), 400
            
            file1 = request.files['img1']
            file2 = request.files['img2']
            
            # Créer fichiers temporaires
            temp1 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp2 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            
            file1.save(temp1.name)
            file2.save(temp2.name)

            # IMPORTANT : fermer avant utilisation
            temp1.close()
            temp2.close()
            
            temp_files = [temp1.name, temp2.name]
            img1_path, img2_path = temp1.name, temp2.name

        elif request.content_type == 'application/json':
            data = request.json
            img1_path = data.get('img1_path')
            img2_path = data.get('img2_path')
            
            if not img1_path or not img2_path:
                return jsonify({"error": "Champs 'img1_path' et 'img2_path' requis"}), 400
        else:
            return jsonify({"error": "Type de contenu non supporté"}), 400
        
        # Vérifier l’existence
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            return jsonify({"error": "Un ou plusieurs fichiers non trouvés"}), 400
        
        # Comparer avec DeepFace
        result = DeepFace.verify(img1_path, img2_path)
        
        # Conversion numpy → types natifs
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

    finally:
        # Nettoyage des fichiers temporaires
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except PermissionError:
                    print(f"Impossible de supprimer {f} (encore utilisé)")

@app.route('/extract_faces', methods=['POST'])
def extract_faces():
    try:
        img_path = None
        temp_file = None
        
        if request.content_type and 'multipart/form-data' in request.content_type:
            if 'img' in request.files:
                file = request.files['img']
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                file.save(temp_file.name)
                img_path = temp_file.name
            else:
                return jsonify({"error": "Aucun fichier 'img' trouvé"}), 400
                
        elif request.content_type == 'application/json':
            data = request.json
            img_path = data.get('img_path')
            if not img_path:
                return jsonify({"error": "Champ 'img_path' manquant"}), 400
        else:
            return jsonify({"error": "Type de contenu non supporté"}), 400
        
        # Extraire les visages
        faces = DeepFace.extract_faces(img_path)
        
        # Convertir en liste (les images sont des arrays numpy)
        faces_list = []
        for i, face in enumerate(faces):
            faces_list.append({
                "face_id": i,
                "confidence": 1.0,  # DeepFace ne retourne pas toujours la confidence
                "region": "detected"  # Information basique
            })
        
        # Nettoyer le fichier temporaire
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        
        return jsonify({
            "faces_detected": len(faces_list),
            "faces": faces_list
        })
        
    except Exception as e:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "message": "DeepFace API is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)