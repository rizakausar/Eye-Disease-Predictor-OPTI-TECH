from flask import Flask, request, redirect, render_template, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'optitech_ultra_secure_789'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///optitech.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Record(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50))
    disease = db.Column(db.String(100))
    confidence = db.Column(db.String(20))
    image_path = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.now) # Stores Date and Time
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load AI Model
model = load_model('models/eyess.h5', compile=False)
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

SOLUTIONS = {
    "Cataract": "Wear sunglasses to protect from UV rays,Eat foods rich in vitamin C and antioxidants (oranges, carrots),Avoid smoking and excessive alcohol,Maintain good lighting while reading.",
    "Diabetic Retinopathy": "Control blood sugar levels strictly,Eat a balanced diet (low sugar, high fiber),Exercise regularly,Monitor blood pressure and cholesterol,Avoid smoking.",
    "Glaucoma": "Eat leafy greens,Exercise regularly (helps eye pressure),Avoid excessive caffeine,Protect eyes from strain (limit screen time),Sleep with head slightly elevated,Safe medical suggestions,Regular eye checkups are very important.",
    "Normal": "No issues detected. Maintain regular checkups and a healthy diet.",
    "Not an Eye Image": "The prediction is not accurate,so the uploaded image is identified as not an eye image."
}

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        uname = request.form.get("username")
        pwd = request.form.get("password")
        cpwd = request.form.get("confirm_password")
        user = User.query.filter_by(email=email).first()
        if not user:
            if pwd != cpwd: return "Passwords do not match!", 400
            user = User(username=uname, email=email, password=pwd)
            db.session.add(user)
            db.session.commit()
        if user.password == pwd:
            login_user(user)
            session['db_unlocked'] = False 
            return redirect(url_for('upload_page'))
        return "Invalid Credentials", 401
    return render_template("login.html")

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    file = request.files["file"]
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    img = Image.open(path).convert("RGB").resize((224, 224))
    x = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(x)
    max_conf = float(np.max(pred))
    disease = CLASS_NAMES[np.argmax(pred)] if max_conf >= 0.60 else "Not an Eye Image"
    conf_pct = f"{round(max_conf * 100, 2)}%"
    new_rec = Record(username=current_user.username, disease=disease, confidence=conf_pct, 
                    user_id=current_user.id, image_path=filename)
    db.session.add(new_rec)
    db.session.commit()
    return redirect(url_for('result', disease=disease, confidence=conf_pct, img=filename))

@app.route("/result")
@login_required
def result():
    disease = request.args.get('disease')
    solution = SOLUTIONS.get(disease, "Consult a specialist.")
    return render_template("result.html", solution=solution)

@app.route("/database")
@login_required
def database():
    if not session.get('db_unlocked'): return redirect(url_for('db_auth'))
    records = Record.query.filter_by(user_id=current_user.id).all()
    return render_template("user_db.html", records=records)

@app.route("/delete-record/<int:rid>")
@login_required
def delete_record(rid):
    rec = Record.query.get(rid)
    if rec and rec.user_id == current_user.id:
        db.session.delete(rec)
        db.session.commit()
    return redirect(url_for('database'))

@app.route("/db-auth", methods=["GET", "POST"])
@login_required
def db_auth():
    if request.method == "POST":
        if request.form.get("password") == current_user.password:
            session['db_unlocked'] = True
            return redirect(url_for('database'))
    return render_template("db_lock.html")

@app.route("/upload")
@login_required
def upload_page(): return render_template("upload.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))

if __name__ == "__main__":
    with app.app_context(): db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)