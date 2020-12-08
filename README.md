# License Plate Reader
<a href="https://herokutrialopencv.herokuapp.com/">Website Link</a>
<br></br>

<div class="img-responsive">
<img src="https://github.com/sauravakolia/License_Plate_Reader/blob/main/static/detections/00c82d64185293a7.jpg" width="300" height="200">
</div>

Download the weights and store in the weights folder.
<br></br>
<a href="https://drive.google.com/file/d/1-3m0nYWJNWNuZSceAktTHWqJ2odQT1YI/view?usp=sharing">Yolov4 Weights</a>

<a href="https://colab.research.google.com/drive/1aJm4yzsulCxjVWUXOPEW0RAJ68MDbYHa?usp=sharing"> Plate Detection Notebook</a>

<a href="https://github.com/sauravakolia/License_Plate_Detector">Plate Detection Repository</a>
<br></br>

# Dataset
* Google Open Image is used for data gathering of plates and their annotations.
* For downloading the data set from <i>Open Image</i> OIDv4 ToolKit is used
* Training dataset includes 1500 images and annotations 
  'python3 main.py downloader --classes Vehicle registration plate --type_csv train -limit 1500'
* Validation dataset includes 700 images and annotations 
   'python3 main.py downloader --classes Vehicle registration plate --type_csv validation -limit 700'
<br></br>

