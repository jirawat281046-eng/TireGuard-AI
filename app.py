print("\n--- STARTING WEB SERVER ---")
import os
# --- SAFETY MODE: FORCE CPU & DISABLE HANGING OPTS ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silent mode for TF

import numpy as np
import random
print("Loading libraries (Numpy, Flask)...")
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

print("Loading TensorFlow... (This can take 30-60s on some machines)")
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Configuration
MODEL_PATH = 'model_resnet50v2_multiclass.h5'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model globally for efficiency
print(f"Loading Multi-class model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process and Predict
            img_array = process_image(filepath)
            
            import time
            start_time = time.time()
            # New Prediction logic for 3 classes
            prediction = model.predict(img_array, verbose=0)
            inference_time = (time.time() - start_time) * 1000 # in ms
            
            # 1. Handle Not Tire (OOD) or Low Confidence (Uncertainty)
            # Index 0: defective, Index 1: good, Index 2: not_tire
            class_idx = np.argmax(prediction[0])
            top_confidence = float(prediction[0][class_idx])
            
            # THRESHOLD: If confidence < 0.75 or it's 'not_tire' category
            if class_idx == 2 or top_confidence < 0.75:
                label = 'ไม่สามารถวิเคราะห์ได้'
                display_confidence = 0.0
                selected_advice = "ไม่สามารถระบุข้อมูลในภาพได้ กรุณาตรวจสอบและถ่ายภาพหน้ายางใหม่อีกครั้งในที่ที่มีแสงสว่าง"
                is_good = False
            else:
                # 2. Handle Confident Tire Prediction
                is_good = (class_idx == 1)
                label = 'ปกติ' if is_good else 'ชำรุด/เสียหาย'
                display_confidence = top_confidence
    
                if not is_good:
                    advices = [
                        "พบความชำรุดหรือรอยเสียหาย: ควรตรวจสอบอย่างละเอียดเพื่อความปลอดภัย",
                        "ตรวจพบโครงสร้างยางผิดปกติ: เสี่ยงต่อการระเบิด ไม่ควรใช้ความเร็วสูง",
                        "แก้มยางมีรอยฉีกขาด: แนะนำให้เปลี่ยนยางทันทีถ้าเป็นไปได้"
                    ]
                else:
                    advices = [
                        "ยางอยู่ในสภาพดี: ควรหมั่นเช็คลมยางสม่ำเสมอทุกๆ 2 สัปดาห์",
                        "ไม่พบสิ่งผิดปกติ: แนะนำให้สลับยางทุก 10,000 กม. เพื่อยืดอายุการใช้งาน",
                        "สภาพดอกยางปกติ: อย่าลืมตรวจสอบหน้ายางหาสิ่งแปลกปลอมสม่ำเสมอ"
                    ]
                selected_advice = random.choice(advices)
             
            return jsonify({
                'label': label,
                'confidence': f"{display_confidence:.2%}",
                'raw_confidence': display_confidence,
                'advice': selected_advice,
                'inference_time_ms': f"{inference_time:.1f} ms",
                'status': 'success'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up: remove the uploaded file after prediction
            if os.path.exists(filepath):
                os.remove(filepath)
                
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Port 7860 is the standard for Hugging Face Spaces
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)
