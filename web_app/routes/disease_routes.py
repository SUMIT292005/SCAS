from flask import Blueprint, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
from web_app.dashboard_helpers import detect_disease_from_image

disease_bp = Blueprint("disease", __name__)

UPLOAD_FOLDER = "web_app/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@disease_bp.route("/disease_detector", methods=["GET", "POST"])
def detect_disease():
    prediction, confidence, image_url = None, None, None

    if request.method == "POST":
        crop = request.form["crop"]
        image_file = request.files["image"]

        if crop and image_file:
            filename = secure_filename(image_file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            image_file.save(file_path)

            prediction, confidence = detect_disease_from_image(file_path, crop)
            image_url = url_for("static", filename=f"uploads/{filename}")

    return render_template("disease_detector.html", 
                           prediction=prediction, 
                           confidence=confidence,
                           image_url=image_url)
