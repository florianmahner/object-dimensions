import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path


app = Flask(__name__, template_folder="templates")

# Configure the folder to store uploaded images

THIS_FOLDER = Path(__file__).parent.resolve()
UPLOAD_FOLDER = os.path.join(THIS_FOLDER, "static", "images")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["STATIC_FOLDER"] = os.path.join(THIS_FOLDER, "static")

# Create the upload folder if it does not exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Only allow certain file types to be uploaded
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


image_files = [f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if allowed_file(f)]

image_files.sort()


@app.route("/", methods=["GET", "POST"])
def index():
    # Get a list of all image files in the upload folder
    image_files = [
        f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if allowed_file(f)
    ]

    image_files.sort()

    # If there are no image files in the folder, return an error message
    if not image_files:
        return "No image files found in the upload folder."

    # Determine the index of the current image based on the 'index' form parameter
    if request.method == "POST":
        if request.form["submit_button"] == "Next":
            index = int(request.form["index"]) + 1
            if index == len(image_files):
                return redirect(url_for("end"))
        else:
            index = int(request.form["index"]) - 1
    else:
        index = 0

    # Load the current image and render the template
    filename = image_files[index]
    filepath = os.path.join("static", "images", filename)

    return render_template("index.html", image_filepath=filepath, index=index)


@app.route("/end")
def end():
    return render_template("end.html")


if __name__ == "__main__":
    app.run()
