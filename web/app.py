from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    request,
    flash,
    session,
    send_from_directory,
)
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    logout_user,
    current_user,
    UserMixin,
)
import os
from datetime import datetime
import numpy as np
import nibabel as nib 
import plotly.graph_objects as go
from segmentor import Segmentor
from segmentor import PATIENTS_PATH
from utils import get_extension

app = Flask(__name__)
app.config["SECRET_KEY"] = "cyber_SAY_2024"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['VIDEO_FOLDER'] = 'static/videos'
app.config["PROFILE_PIC_FOLDER"] = os.path.join(
    app.config["UPLOAD_FOLDER"], "profilePic"
)
app.config["PROFILE_PIC_FOLDER"] = os.path.join(app.root_path, "uploads/profilePic")



app.config["ALLOWED_EXTENSIONS"] = {"nii", "nii.gz"}
app.config["ALLOWED_EXTENSIONS_FOR_FILE"] = {"png", "jpg", "jpeg"}
app.config["ALLOWED_EXTENSIONS_FOR_PROFILE"] = {"png", "jpg", "jpeg"}
bcrypt = Bcrypt(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

niftiSegmentor = Segmentor()


# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    dob = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    medical_info = db.Column(db.String(100), nullable=True)
    recovery_email = db.Column(db.String(150), nullable=True)
    profile_pic = db.Column(db.String(200), nullable=True)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


VIDEO_FOLDER = os.path.join(os.getcwd(), "static", "videos")


@app.route("/video")
def video():
    # Check if user is authenticated
    if current_user.is_authenticated:
        email = current_user.email
        # List all videos starting with user's email
        videos = [f for f in os.listdir(VIDEO_FOLDER) if f.startswith(email)]
        return render_template("video.html", email=email, videos=videos)
    else:
        return (
            "You must be logged in to view this page.",
            403,
        )  # Error if not logged in


# Serve the video files
@app.route("/static/videos/<filename>")
def video_file(filename):
    return send_from_directory(VIDEO_FOLDER, filename)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def allowed_file(filename):
    for extension in app.config["ALLOWED_EXTENSIONS"]:
        if filename.endswith(extension):
            return True
    return False


def allowed_file_for_profile(filename):
    for extension in app.config["ALLOWED_EXTENSIONS_FOR_PROFILE"]:
        if filename.endswith(extension):
            return True
    return False


# Helper function to list uploaded files by the current user
def get_user_uploaded_files():
    user_files = []
    if current_user.is_authenticated:
        for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
            if filename.startswith(current_user.email):  # Users files
                user_files.append(filename)
    return user_files


@app.route("/")
@app.route("/home")
def home():
    return render_template(
        "home.html", email=current_user.email if current_user.is_authenticated else None
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form["full_name"]
        email = request.form["email"]
        password = request.form["password"]
        dob_str = request.form["dob"]
        gender = request.form["gender"]
        medical_info = request.form["medical_info"]
        recovery_email = request.form["recovery_email"]

        dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

        new_user = User(
            full_name=full_name,
            email=email,
            password=hashed_password,
            dob=dob,
            gender=gender,
            medical_info=medical_info,
            recovery_email=recovery_email,
        )

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['user_email'] = email 
            login_user(user)
            return redirect(url_for("home"))
        else:
            flash("Login unsuccessful. Please check email and password", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        axis = request.form["axis"]
        if (
            "ed_file" not in request.files
            or "es_file" not in request.files
            or "cine_file" not in request.files
        ):
            flash("One or more files are missing.", "danger")
            return redirect(request.url)

        ed_file = request.files["ed_file"]
        es_file = request.files["es_file"]
        cine_file = request.files["cine_file"]

        files = {"ED": ed_file, "ES": es_file, "CINE": cine_file}
        for file_type, file in files.items():
            if file.filename == "":
                flash(f"No selected file for {file_type}", "danger")
                return redirect(request.url)

            if file and allowed_file(file.filename):
                user_identifier = current_user.email
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                new_filename = f"{user_identifier}_{timestamp}_{axis}_{file_type}.{get_extension(file.filename, app.config['ALLOWED_EXTENSIONS'])}"
                file.save(os.path.join(PATIENTS_PATH, new_filename))
        niftiSegmentor.generate_patient_info(
            f"{user_identifier}_{timestamp}_{axis}_{{}}.{get_extension(file.filename, app.config['ALLOWED_EXTENSIONS'])}",
            f"{user_identifier}_{timestamp}_{axis}",
            axis,
        )
        flash("Files uploaded and renamed successfully!", "success")
        return redirect(url_for("upload"))

    return render_template("upload.html")


@app.route("/profile")
@login_required
def profile():
    # Use the default profile picture if none is set
    profile_pic_filename = current_user.profile_pic or "49.png"

    # Generate the profile picture URL
    profile_pic_url = url_for('uploaded_file', filename=profile_pic_filename)

    uploaded_files = get_user_uploaded_files()

    return render_template(
        "profile.html",
        user=current_user,
        uploaded_files=uploaded_files,
        profile_pic=profile_pic_url,
    )



from flask import send_from_directory

@app.route('/uploads/profilePic/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["PROFILE_PIC_FOLDER"], filename)



@app.route("/download/<filename>")
@login_required
def download_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/upload_photo", methods=["POST"])
@login_required
def upload_photo():
    if "profile_photo" not in request.files:
        flash("No file part", "danger")
        return redirect(url_for("profile"))

    file = request.files["profile_photo"]
    if file.filename == "":
        flash("No selected file", "danger")
        return redirect(url_for("profile"))


    if file and allowed_file_for_profile(file.filename):
        filename = f"{current_user.email}.png"
        file_path = os.path.join(app.config["PROFILE_PIC_FOLDER"], filename)
        file.save(file_path)

        current_user.profile_pic = filename
        db.session.commit()

        flash("Profile photo uploaded successfully!", "success")

    else:
        flash("Invalid file type", "danger")

    return redirect(url_for("profile"))



@app.route("/change_password", methods=["POST"])
@login_required
def change_password():
    new_password = request.form["new_password"]
    hashed_password = bcrypt.generate_password_hash(new_password).decode("utf-8")
    current_user.password = hashed_password
    db.session.commit()
    flash("Password changed successfully!", "success")
    return redirect(url_for("profile"))


@app.route("/delete_account", methods=["POST"])
@login_required
def delete_account():
    db.session.delete(current_user)
    db.session.commit()
    logout_user()
    flash("Your account has been deleted", "info")
    return redirect(url_for("home"))


# ------------------------------------------------------

@app.route("/upload_nifti", methods=["GET", "POST"])
def upload_nifti():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file and (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            return redirect(url_for("view_nifti", filename=file.filename))
        else:
            return "Error: Only NIfTI files (.nii, .nii.gz) are allowed."

    return render_template("upload_nifti.html")


from nifti_viewer import create_nifti_visualization
@app.route("/view_nifti/<filename>")
def view_nifti(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    error_message, graph_html = create_nifti_visualization(filepath)
    
    if error_message:
        return error_message
    
    return render_template("viewer.html", graph_html=graph_html)


@app.route("/3d-segmentation")
def segmentation():
    return render_template("3d_segmentation.html")

#---------------------------------------------------------------------------------------
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for, flash
import os
import nibabel as nib
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from datetime import datetime
import matplotlib
matplotlib.use('Agg')



@app.route('/create_animation', methods=['POST'])
def create_animation():
    if 'nifti_file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['nifti_file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        email = current_user.email
        create_animation_from_nifti(filepath, filename, email)

        return redirect(url_for('video'))


def create_animation_from_nifti(nifti_path, filename, email):
    img = nib.load(nifti_path)
    data = img.get_fdata()

    slice_data = data[:, :, data.shape[2] // 2, :]

    frames_folder = 'frames/'
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    num_frames = slice_data.shape[2]
    for i in range(num_frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(slice_data[:, :, i], cmap='gray', vmin=slice_data.min(), vmax=slice_data.max())
        ax.set_title(f'Frame {i+1}')
        
        frame_path = os.path.join(frames_folder, f'frame_{i:03d}.png')
        plt.savefig(frame_path)
        plt.close(fig)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    animation_filename = f"{email}_{timestamp}_{os.path.splitext(filename)[0]}.mp4"
    animation_path = os.path.join('static/videos/', animation_filename)

    image_files = [os.path.join(frames_folder, f'frame_{i:03d}.png') for i in range(num_frames)]
    clip = ImageSequenceClip(image_files, fps=15)
    clip.write_videofile(animation_path, codec='libx264')

    for file in image_files:
        os.remove(file)


#------------------------------------------------------------------------------------------------------

@app.route('/interactive')
def interactive():
    return render_template('interactive.html')

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
