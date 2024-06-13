import os
import glob
import random
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    make_response,
)
from pathlib import Path
import uuid
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def generate_id():
    id = str(uuid.uuid4())
    time = datetime.now().strftime("%d-%m-%Y-%H%M%S")
    session_id = time + "_" + id
    return session_id


app = Flask(__name__)
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(days=7)

# Configure the folder to store uploaded images
THIS_FOLDER = Path(__file__).parent.resolve()
UPLOAD_FOLDER = os.path.join(THIS_FOLDER, "static", "images")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["STATIC_FOLDER"] = os.path.join(THIS_FOLDER, "static")

# Create the upload folder if it does not exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(os.path.join(THIS_FOLDER, "results")):
    os.makedirs(os.path.join(THIS_FOLDER, "results"))


@app.route("/consent", methods=["GET", "POST"])
def consent():
    return render_template("consent.html")


@app.route("/data", methods=["GET", "POST"])
def data():
    return render_template("data.html")


@app.route("/instructions", methods=["GET", "POST"])
def instructions():
    return render_template("instructions.html")


@app.route("/index", methods=["GET", "POST"])
def show_index():
    params = np.load(
        os.path.join(THIS_FOLDER, "results", f"{session['session_id']}.npz")
    )

    image_filepath = params["filepath"]
    index = params["index"]

    return render_template("index.html", image_filepath=image_filepath, index=index)


@app.route("/", methods=["GET", "POST"])
def index():
    session_id = session.get("session_id")
    session.permanent = True

    if not session_id:
        session_id = generate_id()
        session["session_id"] = session_id

    image_files = session.get("image_files")
    if not image_files:
        image_files = glob.glob(
            os.path.join(UPLOAD_FOLDER, "**", "*.jpg"), recursive=True
        )
        image_files = [os.path.basename(f) for f in image_files]
        image_files.remove("instructions.jpg")
        image_files = sorted(image_files, key=lambda x: int(x.split("_")[0]))
        random.shuffle(image_files)  # Each participant sees a different order
        session["image_files"] = image_files

    if not session.get("consent_given"):
        session["consent_given"] = True
        return redirect(url_for("consent", session_id=session_id))

    if not session.get("data_given"):
        session["data_given"] = True
        return redirect(url_for("data", session_id=session_id))

    # Check if consent has been given
    if not session.get("instructions_given"):
        session["instructions_given"] = True
        return redirect(url_for("instructions", session_id=session_id))

    if "output" not in session:
        descriptions = [f"description_{i}" for i in range(1, 6)]
        session["output"] = pd.DataFrame(
            columns=["image_file", "dim", *descriptions, "interpretability"]
        ).to_json()
    output = pd.read_json(session["output"])

    # If there are no image files in the folder, return an error message
    if not image_files:
        return "No image files found in the upload folder."

    # Retrieve the current index from the URL query parameter or session, or set it to 0 if it doesn't exist
    if request.args.get("index"):
        index = int(request.args["index"])

    else:
        index = session.get("index", 0)

    # Determine the index of the current image based on the 'index' form parameter
    if request.method == "POST":
        if request.form["submit_button"] == "Next":
            descriptions = request.form.get("descriptions", "")
            interpretability = request.form.get("interpretability", 0)

            image_files = session["image_files"]
            fname = image_files[index]
            # Split the descriptions with a comma
            descriptions = descriptions.split(",")
            descriptions = descriptions + [""] * (5 - len(descriptions))

            # Add the descriptions to the output dataframe at the index
            dim = fname.split("_")[0]
            output.loc[index] = [fname, dim, *descriptions, interpretability]

            # Save the results to a CSV file
            try:
                output.to_csv(
                    os.path.join(
                        THIS_FOLDER, "results", f"{session['session_id']}.csv"
                    ),
                    index=False,
                )
            except:
                pass

            session["output"] = output.to_json()

            index += 1

            if index == len(image_files):
                if index == len(image_files):
                    return redirect(url_for("end"))

    filename = image_files[index]

    filepath = os.path.join(
        "https://fmahner.pythonanywhere.com/static/images/", filename
    )

    session["filepath"] = filepath
    session["index"] = index

    np.savez(
        os.path.join(THIS_FOLDER, "results", f"{session['session_id']}.npz"),
        image_files=image_files,
        index=index,
        filepath=filepath,
    )

    return redirect(url_for("show_index"))


@app.route("/end")
def end():
    return render_template("end.html")


if __name__ == "__main__":
    app.run(port=2500, debug=True)
