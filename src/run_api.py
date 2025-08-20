from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import json
import os
import tempfile
from flask_cors import CORS
import psycopg2

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
CORS(app, origins=["http://localhost:5173"])

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



def get_user_image_bytes(user_id):
    conn = psycopg2.connect(
        dbname="users_db",
        user="postgres",
        password="root",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT image 
        FROM user_image 
        JOIN user_entity ON user_image.user_id = user_entity.id 
        WHERE user_entity.id = %s AND user_image.type = %s
    """, (user_id, "PROFILE"))
    result = cur.fetchone()
    conn.close()
    if result:
        return result[0]
    return None

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

@app.route('/verify', methods=['POST'])
def verify_faces():
    temp_files = []
    try:
        # Récupérer image 1 envoyée par form-data
        if 'img' not in request.files:
            return jsonify({"error": "Fichier 'img' requis"}), 400

        file1 = request.files['img']
        temp1 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file1.save(temp1.name)
        temp1.close()
        temp_files.append(temp1.name)
        img1_path = temp1.name

        # Vérifier que le fichier existe
        if not os.path.exists(img1_path):
            return jsonify({"error": "img1 introuvable"}), 400

        # Récupérer userId
        user_id = request.form.get('userId') or request.args.get('userId')
        if not user_id:
            return jsonify({"error": "Paramètre 'userId' manquant"}), 400

        # Récupérer image 2 depuis DB
        image2_bytes = get_user_image_bytes(user_id)
        if not image2_bytes:
            return jsonify({"error": f"Aucune image trouvée pour l'utilisateur {user_id}"}), 404

        temp2 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp2.write(image2_bytes)
        temp2.close()
        temp_files.append(temp2.name)
        img2_path = temp2.name

        if not os.path.exists(img2_path):
            return jsonify({"error": "img2 introuvable"}), 400

        # DeepFace verify
        result = DeepFace.verify(img1_path, img2_path)
        result = convert_numpy(result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Nettoyage des fichiers temporaires
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

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