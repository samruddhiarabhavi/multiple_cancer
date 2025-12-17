# 🩺 Multiple Cancer Prediction System

## 🔗 Live Demo

🚀 [https://multiple-cancer.onrender.com/](https://multiple-cancer.onrender.com/)

A **web-based machine learning application** that predicts the risk of **multiple cancers** (such as **Breast Cancer**, **Lung Cancer**, etc.) based on user-provided medical parameters. This project is built to support **early risk assessment** and demonstrate the use of **ML models in healthcare applications**.

---

## 🚀 Features

* 🔍 Predicts **multiple cancer types** from a single platform
* 🧠 Uses **trained Machine Learning models**
* 🌐 Web-based interface (Flask)
* 🔐 Login & Signup system (if enabled)
* 📊 Clean and user-friendly UI
* ☁️ Deployable on cloud platforms (Render)

---

## 🏗️ Project Structure

```
multiple_cancer_prediction/
│── app.py / main.py        # Main application file
│── models/                # Trained ML models (.pkl)
│── templates/             # HTML files (Flask)
│── static/                # CSS, JS, images
│── dataset/               # Training datasets
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
```

---

## 🧪 Cancer Types Supported

* 🧬 Breast Cancer Prediction
* 🫁 Lung Cancer Prediction
* ➕ More cancers can be added easily

---

## ⚙️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript / Streamlit
* **Backend:** Python, Flask / Streamlit
* **Machine Learning:** Scikit-learn
* **Data Handling:** NumPy, Pandas
* **Deployment:** Gunicorn, Render

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/multiple-cancer-prediction.git
cd multiple-cancer-prediction
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

#### Flask:

`bash
python app.py


## 🌐 Deployment Notes

* Ensure **Gunicorn** is included in `requirements.txt`
* Use the correct start command:

```bash
gunicorn app:app
```

* App must bind to:

```python
app.run(host="0.0.0.0", port=PORT)
```

---

## 📸 Screenshots
![WhatsApp Image 2025-12-07 at 7 25 52 PM](https://github.com/user-attachments/assets/41886daa-f0bc-49ae-85ca-f846fe18131f)


![WhatsApp Image 2025-12-07 at 7 25 52 PM](https://github.com/user-attachments/assets/01a1c8f0-555d-4870-9f97-90d3830452b3)

---

## 🎯 Use Cases

* Academic & Mini Projects
* Healthcare ML Demonstrations
* Resume / Portfolio Projects

---

## 🔮 Future Enhancements

* 🔐 Role-based authentication
* 📈 Model performance visualization
* 🧠 Deep Learning models (CNN for X-rays)
* 📱 Mobile-friendly UI

---

## ⚠️ Disclaimer

This project is **for educational purposes only**. It should **not be used for real medical diagnosis**.

---

## 👩‍💻 Author

**Samruddhi Arabhavi**
CSE Student | ML & Full Stack Enthusiast

---

## ⭐ Support

If you like this project, please **⭐ star the repository** and share it!


