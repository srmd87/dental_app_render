from flask import Flask, request, render_template, jsonify
import os
import requests
from tensorflow import keras
from PIL import Image
import numpy as np

app = Flask(__name__)

model_path = "dental_model.keras"
hf_url = "https://huggingface.co/srmd87/dental-model-ai/resolve/main/dental_model.keras"

if not os.path.exists(model_path):
    print("⬇️ تحميل النموذج من Hugging Face...")
    r = requests.get(hf_url)
    if r.status_code == 200:
        with open(model_path, "wb") as f:
            f.write(r.content)
        print("✅ تم تحميل النموذج بنجاح.")
    else:
        raise Exception(f"❌ فشل تحميل النموذج. رمز الحالة: {r.status_code}")

model = keras.models.load_model(model_path)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join("static", file.filename)
            os.makedirs("static", exist_ok=True)
            file.save(filepath)
            img_array = preprocess_image(filepath)
            result = model.predict(img_array)
            predicted_class = int(np.argmax(result))
            prediction = f"Predicted class: {predicted_class}"
            return jsonify({"prediction": prediction})
    return render_template("index.html", prediction=prediction)
