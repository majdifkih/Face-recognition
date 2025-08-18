from fastapi import FastAPI
from deepface import DeepFace
import psycopg2
from io import BytesIO
from fastapi import HTTPException

app = FastAPI()

def get_user_image(user_id: int):
    """Récupère l'image de l'utilisateur depuis la base de données."""
    conn = psycopg2.connect(
        dbname="users_db",
        user="postgres",
        password="root",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute("SELECT image FROM user_image WHERE user_id = %s AND type = %s", (user_id, "PROFILE"))
    result = cur.fetchone()
    conn.close()
    if result:
        return result[0]  # Données binaires de l'image
    return None

@app.post("/verify/")
async def verify_user(user_id: int, image_data: bytes):
    """Vérifie si l'image capturée correspond à l'image de l'utilisateur."""
    user_image = get_user_image(user_id)
    if user_image is None:
        raise HTTPException(status_code=404, detail="Image de l'utilisateur non trouvée")

    # Analyse de l'image capturée
    try:
        result = DeepFace.verify(
            img1_path=BytesIO(user_image),
            img2_path=BytesIO(image_data),
            model_name="VGG-Face",
            detector_backend="opencv",
            distance_metric="cosine"
        )
        return {"verification": result["verified"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
