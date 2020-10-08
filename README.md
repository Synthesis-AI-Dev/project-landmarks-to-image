# project-landmarks-to-image
This script projects facial landmarks onto the RGB image for visualization.

![Method](images/landmarks_sample.png)

## Usage
The arguments for the script are passed via the config file. All the parameters in the
 config file can be overriden from the command line. 
 
Example: Visualize landmarks for all images in dir `samples/` and save output to same dir.
 
 ```shell script
python project_landmarks_on_images.py dir.input:samples/ dir.output:null
```

## Installation
The script requires python 3. Tested on Python 3.8.3. Install pip packages via:

```shell script
pip install -r requirements.txt
```