"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template,request

from model import *


from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

import os
import sys

from flask import Flask

from flask_mysqldb import MySQL


app = Flask(__name__)

app.config['MYSQL_HOST'] = 'us-cdbr-east-02.cleardb.com'
app.config['MYSQL_USER'] = 'b065bb4a6c47b3'
app.config['MYSQL_PASSWORD'] = '592b2369'
app.config['MYSQL_DB'] = 'heroku_8bb5027e2cb7476'

mysql = MySQL(app)
                       



UPLOAD_FOLDER="static/uploads"
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'JPG'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
#app.config['DETECTION_FOLDER'] = DETECTION_FOLDER

@app.route("/")
def index():
  return render_template("index.html")

# @app.route("/about")
# def about():
#   return render_template("about.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      # create a secure filename
      filename = secure_filename(f.filename)
      print(filename)
      # save file to /static/uploads
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      print(filepath)
      f.save(filepath)

      image_path=os.path.join(filepath,filename)
      plate_no=plate_dtection(filepath,filename)


      onwer_name=request.form.get("uname")

      mycur=mysql.connection.cursor()
      sql="INSERT INTO Owner_Data(name,car_plate) VALUES (%s,%s) " 
      value=onwer_name,plate_no
      mycur.execute(sql, value)
      mysql.connection.commit()
      mycur.close()


      return render_template("uploaded.html", display_detection = filename, fname = filename, onwer_name=onwer_name)      

@app.route("/data")
def user_data():

  mycursor=mysql.connection.cursor()
  mycursor.execute("SELECT id,name,car_plate FROM Owner_Data")
  x=mycursor.fetchall()
  mysql.connection.commit()
  mycursor.close()
  
  return render_template('data.html',posts=x)




if __name__ == '__main__':
   app.run()

