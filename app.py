from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
# import sys
# sys.path.append('../')
import json
from utils import *
import os
from flask import jsonify
app = Flask(__name__)
dropzone = Dropzone(app)


app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = "do"

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/static/images'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


@app.route('/', methods=['GET', 'POST'])
def index():
    
    # set session for image results
    # if "file_urls" not in session:
    #     session['file_urls'] = []
    # if "all_faces" not in session:
    #     session['all_faces'] = []
    # list to hold our uploaded image urls
    # file_urls = session['file_urls']

    # handle image upload from Dropszone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            
            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename    
            )
    # if request.method=="POST"
    #     all_faces=load_data("./uploads")
    #     result=do_clusterV2(all_faces)
    #     print(result[0]["images"])
    #     return jsonify(result)
    # if request.method=="POST":
    #     all_faces=load_data("./uploads")
    #     result=do_clusterV2(all_faces)
    #     print(result)
            # append image urls
            # file_urls.append(photos.url(filename))
            
        # session['file_urls'] = file_urls
        
    #     # session["all_faces"]=all_faces
    #     return "uploading..."
    # return dropzone template on GET request    
    return render_template('index.html')

@app.route('/do',methods=["GET","POST"])
def do():
    res=[ ]
    all_faces=load_data("./static/images")
    
    result=do_cluster(all_faces)
    for i in range(len(result)):
        res.append({"cluster":result[i]["images"]})
    # return jsonify({"hi":res})
   

    return render_template("result.html",images=res)

    # return jsonify("j")        
        # filename = os.path.join(app, 'data1')
        # print(os.listdir(filename))
       # data=load_data("./data1")

@app.route('/results')
def results():
    # all_faces=load_data("./uploads")
    # result=do_clusterV2(session["all_faces"])
    # print(session)
    # print(os.listdir("./uploads"))
    # redirect to home if no images to display
    # if "file_urls" not in session or session['file_urls'] == []:
    #     return redirect(url_for('index'))
        
    # set the file_urls and remove the session variable
    # file_urls = session['file_urls']
    # session.pop('file_urls', None)
    
    
    return render_template('results.html')
if __name__ == '__main__':
    app.run(threaded=False)
