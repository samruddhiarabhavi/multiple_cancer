import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Torch for CNN
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import secrets
from datetime import datetime
from openai import OpenAI

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(16)
print(f"[INFO] Using secret key: {app.secret_key[:8]}***")

DB_NAME = "users.db"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "bmp"}

def allowed_file(filename, allowed_set=ALLOWED_IMAGE_EXT):
    if not filename:
        return False
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set

# -----------------------------
# DB init
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS health_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            name TEXT,
            age INTEGER,
            gender TEXT,
            blood_group TEXT,
            notes TEXT,
            xray TEXT,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Model paths - update if needed
# -----------------------------
STRUCT_MODEL_PATH = r"C:\Users\HP\OneDrive\Desktop\multiple cancer\multiple_cancer\lung cancer\lung_cancer\model\model.joblib"
SCALER_PATH = r"C:\Users\HP\OneDrive\Desktop\multiple cancer\multiple_cancer\lung cancer\lung_cancer\model\scaler.joblib"
CNN_MODEL_PATH = r"C:\Users\HP\OneDrive\Desktop\multiple cancer\multiple_cancer\lung cancer\lung_cancer_cnn_pytorch.pt"

# -----------------------------
# Load structural model + scaler
# -----------------------------
try:
    structural_model = joblib.load(STRUCT_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[INFO] Structural model and scaler loaded.")
except Exception as e:
    structural_model = None
    scaler = None
    print(f"[WARN] Could not load structural model/scaler: {e}")

# -----------------------------
# Build CNN
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, img_size=150):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (img_size//8) * (img_size//8), 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = SimpleCNN()

try:
    state = torch.load(CNN_MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        cnn_model.load_state_dict(state["state_dict"])
    else:
        try:
            cnn_model.load_state_dict(state)
        except Exception:
            # maybe saved full model object
            cnn_model = state
    cnn_model.to(device)
    cnn_model.eval()
    print("[INFO] CNN model loaded.")
except Exception as e:
    print(f"[WARN] Could not load CNN model: {e}")
    cnn_model = SimpleCNN().to(device)
    cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# -----------------------------
# System prompt and chat history
# -----------------------------
SYSTEM_PROMPT = (
    "You are a medical AI assistant for cancer prediction and health guidance.\n"
    "Explain results simply, never give a diagnosis, always recommend consulting a doctor.\n"
    "Keep responses short, supportive and non-alarming."
)

# We'll keep a server-wide minimal history list (for production, persist per-user)
chat_history = [
    {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]}
]

# -----------------------------

# Initialize OpenRouter client
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
# Routes (auth, pages, predictions)
# -----------------------------
@app.route("/")
def dashboard():
    return render_template("dash.html", username=session.get("user"))

# Other pages
@app.route("/lung_cancer")
def lung_cancer():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("lung_cancer.html", username=session["user"])

@app.route("/breast_cancer")
def breast_cancer():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("breast_cancer.html", username=session["user"])

@app.route("/skin_cancer")
def skin_cancer():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("skin_cancer.html", username=session["user"])

@app.route("/blood_cancer")
def blood_cancer():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("blood_cancer.html", username=session["user"])

@app.route("/yoga")
def yoga():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("yoga_plan.html", username=session["user"])

@app.route("/hospital-map")
def hospital():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("map.html")

@app.route("/resources")
def resources():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("resources.html", username=session.get("user"))

# Health tracker
@app.route("/healthtracker", methods=["GET", "POST"])
def healthtracker():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        gender = request.form.get("gender")
        blood_group = request.form.get("blood_group")
        notes = request.form.get("notes")

        xray_file = request.files.get("xray")
        filename = None
        if xray_file and xray_file.filename:
            if not allowed_file(xray_file.filename):
                flash("File type not allowed.", "danger")
                return redirect(url_for("healthtracker"))
            filename = datetime.now().strftime("%Y%m%d%H%M%S_") + secure_filename(xray_file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            xray_file.save(filepath)

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            INSERT INTO health_records (username, name, age, gender, blood_group, notes, xray, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session["user"], name, age, gender, blood_group, notes, filename, datetime.now().strftime("%Y-%m-%d %H:%M")))
        conn.commit()
        conn.close()

        flash("Record added successfully!", "success")
        return redirect(url_for("healthtracker"))

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, username, name, age, gender, blood_group, notes, xray, date FROM health_records WHERE username = ?", (session["user"],))
    rows = c.fetchall()
    conn.close()

    records = []
    for row in rows:
        records.append({
            "id": row[0],
            "username": row[1],
            "name": row[2],
            "age": row[3],
            "gender": row[4],
            "blood_group": row[5],
            "notes": row[6],
            "xray": row[7],
            "date": row[8]
        })

    return render_template("healthtracker.html", username=session["user"], records=records)

# -----------------------------
# Auth routes
# -----------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        if not username or not email or not password:
            flash("All fields required!", "danger")
            return redirect(url_for("signup"))
        hashed = generate_password_hash(password)
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed))
            conn.commit()
            conn.close()
            flash("Signup successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists!", "danger")
            return redirect(url_for("signup"))
        except Exception as e:
            flash(f"Error: {e}", "danger")
            return redirect(url_for("signup"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()
        # user tuple: (id, username, email, password)
        if user and check_password_hash(user[3], password):
            session["user"] = user[1]
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials!", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("dashboard"))

# -----------------------------
# Prediction routes
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_image():
    if "user" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        img = Image.open(file).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = cnn_model(img)
            if isinstance(out, torch.Tensor):
                output = out.cpu().numpy().squeeze()
                # ensure scalar
                if isinstance(output, np.ndarray):
                    output = float(output.flatten()[0])
                else:
                    output = float(output)
            else:
                output = float(np.array(out).squeeze())

        # clamp and convert to probability if necessary
        output = max(0.0, min(1.0, float(output)))
        pred = "Cancer Detected" if output > 0.5 else "No Cancer"
        confidence = f"{output*100:.2f}%"
        return jsonify({"prediction": pred, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_structural", methods=["POST"])
def predict_structural():
    if "user" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    if structural_model is None or scaler is None:
        return jsonify({"error": "Structural model/scaler not loaded"}), 500
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        df = pd.read_csv(file)
        if df.shape[0] == 0:
            return jsonify({"error": "CSV has no rows"}), 400
        features = scaler.transform(df)
        preds = structural_model.predict(features)
        probs = structural_model.predict_proba(features)[:,1]
        results = []
        for i, (p, prob) in enumerate(zip(preds, probs)):
            results.append({"row": i+1, "prediction": "Cancer Detected" if int(p)==1 else "No Cancer", "confidence": f"{prob*100:.2f}%"})
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    if "user" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    if structural_model is None or scaler is None:
        return jsonify({"error": "Structural model/scaler not loaded"}), 500
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload received"}), 400
    try:
        # NOTE: Order of features matters. Ensure client sends in the right order.
        features = np.array([list(data.values())]).astype(float)
        features = scaler.transform(features)
        pred = structural_model.predict(features)[0]
        prob = structural_model.predict_proba(features)[0][1]
        return jsonify({"prediction": "Cancer Detected" if int(pred)==1 else "No Cancer", "confidence": f"{prob*100:.2f}%"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Chatbot page + API
# -----------------------------
@app.route("/chatbot")
def chatbot():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("chatbot.html", username=session["user"])


chat_history = [
    {"role": "system", "content": "You are a medical AI assistant for cancer guidance."}
]

@app.route("/chat", methods=["POST"])
def chat():
    if "user" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    chat_history.append({"role": "user", "content": user_message})

    # Keep history short
    MAX_HISTORY = 8
    if len(chat_history) > MAX_HISTORY:
        chat_history[:] = [chat_history[0]] + chat_history[-(MAX_HISTORY-1):]

    try:
        response = openrouter_client.chat.completions.create(
            model="openai/gpt-3.5-turbo",  # OpenRouter supports OpenAI models
            messages=chat_history,
            temperature=0.4
        )

        reply = response.choices[0].message.content
        chat_history.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run
# -----------------------------
if __name__ == "__main__":
    # debug True is fine for development; set to False for production
    app.run(debug=True)
