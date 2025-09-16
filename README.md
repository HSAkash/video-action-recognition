# video-action-recognition
## Create environment
```
python -m venv env
```
## Activate environment
``` sh
sorce env/bin/activate
```
::CMD
```
.\.env\Scripts\activate.bat
```
### PowerShell
```
.\.env\Scripts\Activate.ps1
```
## Install requirements
```
pip intall -r requirements.txt
```
##

* Copy video dataset to dataset/VA-dataset
<pre>
│  
├─config
├─dataset
|  ├─VA-dataset
|      ├─class_01
|      |   ├─ 01.mp4
|      |
|      ├─class_02
|      |    ├─01.mp4
|      |    ├─02.mp4
|  
├─src
|   
</pre>


## For training 
```
python main.py
```