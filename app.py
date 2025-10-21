from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import os
import sqlite3

# --- Flask app setup ---
app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load trained model ---
MODEL_PATH = "best_heavy_qv_xception11.h5"
model = load_model(MODEL_PATH)

# --- Class names ---
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --- Training data distribution from classification report ---
training_distribution = {
    "Buildings": 14.57,
    "Forest": 15.80,
    "Glacier": 18.43,
    "Mountain": 17.50,
    "Sea": 17.00,
    "Street": 16.70
}

# --- Database setup ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    image_name TEXT,
                    predicted_class TEXT,
                    confidence REAL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )''')
    conn.commit()
    conn.close()

init_db()

# --- Home ---
@app.route('/')
def home():
    return render_template('home.html', logged_in='user_id' in session)

# --- Register ---
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_pw = generate_password_hash(password)
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username,password) VALUES (?,?)", (username, hashed_pw))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return "Username already exists"
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')

# --- Login ---
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return "Invalid credentials"
    return render_template('login.html')

# --- Logout ---
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# --- Index (Upload) ---
@app.route('/index')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# --- Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds) * 100)

        user_id = session.get('user_id')
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''INSERT INTO predictions (user_id, image_name, predicted_class, confidence)
                     VALUES (?,?,?,?)''', (user_id, file.filename, predicted_class, confidence))
        conn.commit()

        c.execute("SELECT image_name, confidence FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 5", (user_id,))
        history = c.fetchall()
        conn.close()

        labels = [h[0] for h in history][::-1]
        data = [h[1] for h in history][::-1]

        return render_template(
            'result.html',
            filename=file.filename,
            prediction=predicted_class,
            confidence=f"{confidence:.2f}%",
            labels=labels,
            data=data
        )

# --- Serve uploaded images ---
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# --- Performance ---
@app.route('/performance')
def performance():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT predicted_class, confidence FROM predictions WHERE user_id=?", (user_id,))
    data = c.fetchall()
    conn.close()

    total = len(data)
    avg_conf = round(sum([d[1] for d in data])/total,2) if total>0 else 0

    return render_template('performance.html', total=total, avg_conf=avg_conf, data=data, training_distribution=training_distribution)

# --- Charts ---
@app.route('/charts')
def charts():
    training_distribution = {
        'Buildings': 14.57,
        'Forest': 15.80,
        'Glacier': 18.43,
        'Mountain': 17.50,
        'Sea': 17.00,
        'Street': 16.70
    }
    return render_template('charts.html', training_distribution=training_distribution)

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
