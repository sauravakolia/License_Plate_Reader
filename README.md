# License Plate Reader

The application reads the **License Plate** of the vehicles.
* It uses the combination of Object Detection and OCR techniques.
* It uses Object detection for detecting the License Plate from the vehicles image
* The detected plate is then pass to the OCR algorithm for further reading the License Plate number.
* This application is further deployed on the public server using Heroku </br>


Link to public server  
<a href="https://herokutrialopencv.herokuapp.com/">https://herokutrialopencv.herokuapp.com/</a>
<br></br>

<div class="img-responsive">
<img src="https://github.com/sauravakolia/License_Plate_Reader/blob/main/static/detections/00c82d64185293a7.jpg" width="500" height="300">
</div>

Download the weights and store in the weights folder.
<br></br>
<a href="https://drive.google.com/file/d/1-3m0nYWJNWNuZSceAktTHWqJ2odQT1YI/view?usp=sharing">Yolov4 Weights</a>

<a href="https://colab.research.google.com/drive/1aJm4yzsulCxjVWUXOPEW0RAJ68MDbYHa?usp=sharing"> Plate Detection Notebook</a>

<a href="https://github.com/sauravakolia/License_Plate_Detector">Plate Detection Repository</a>

<a href="https://colab.research.google.com/drive/1VW3-fjaZs_qtRQjUXIAajpNDRTQw_zSA?usp=sharing">OCR Word Trainer Notebook</a>
<br></br>

# Dataset
* Google Open Image is used for data gathering of plates and their annotations.
* For downloading the data set from <i>Open Image</i> OIDv4 ToolKit is used
* Training dataset includes 1500 images and annotations</br>
  `python main.py downloader --classes Vehicle_registration_plate --type_csv train --limit 5000`<br></br>
* Validation dataset includes 700 images and annotations</br>
   `python main.py downloader --classes Vehicle_registration_plate --type_csv validation --limit 1000`
<br></br>

* For Character recogination <a href="http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/">Chars74K dataset</a> is used.
  * Dataset contains 74K images of 64 classes (0-9, A-Z, a-z). The dataset includes characters obtained from natural images (7,705), hand-drawn characters (3,410) and     synthesised characters from computer fonts (62,992).
  * For this project synthesized characters are used


