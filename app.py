import os
from model import Model
from werkzeug.utils import secure_filename
from flask import Flask, render_template, redirect,request,url_for, flash

ALLOWED_FILES = {'jpg','png','jpeg'}
UPLOAD_FOLDER = 'uploaded/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and  filename.rsplit('.',1)[1].lower() in ALLOWED_FILES

@app.route("/")
def Starting():
    return render_template("index.html")

@app.route("/model",methods=['GET', 'POST'])
def PageModelFlask():
    if request.method =='POST':
        if 'file' not in request.files:
            flash('on file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('no selected files')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(file_path)
            result = Model(file_path)
            os.remove(file_path)
            return render_template("workpage.html",output=result)
    return render_template("workpage.html")

if __name__ == '__main__':
    app.run(debug=True)
