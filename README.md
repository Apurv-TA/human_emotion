# Human audio emotion prediction
The purpose of this project is to create an audio emotion detection model which can classify the emotion of the user on the basis of the audio.

## To excute the script with default inputs
  > python __main__.py

## You can also used add arguments which can be observed by using
  > python __main__.py -h

## The result of which is given below
  > python __main__.py [-h] [--log-level LOG_LEVEL] [--log-path LOG_PATH] [--no-console-log] [--data DATA] [--save SAVE] [-v {1,2,3}]
  
### optional arguments:  
> &ensp;&ensp;-h, --help            show this help message and exit  
> &ensp;&ensp;--log-level LOG_LEVEL  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Specify the log level(Default is DEBUG)  
> &ensp;&ensp;--log-path LOG_PATH   Specify the file to which log need to be saved: Default: No file created  
> &ensp;&ensp;--no-console-log      whether or not to write logs to the console: Default: write logs to console  
> &ensp;&ensp;--data DATA           The location where data is to be saved  
> &ensp;&ensp;--save SAVE           Location where artifacts are to be stored  
> &ensp;&ensp;-v {1,2,3}, --verbosity {1,2,3}  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Show the output of Grid Search  

## This package is also uploaded at testpypi and can be downloaded and used using command:
  > pip install -i https://test.pypi.org/simple/ audio-emotion-ApurvTA

After which you can use it simply by using:
  > import audio_emotion

## Using docker to test the model:

### Method 1: Create the docker image and run the container

You can build the image from the docker file using command:  
  > docker build -t project/speech

To run the docker file use:
  > docker run -it --name speech project/speech


it will create a container with all the relevant data copied and the environment created for
our purpose


### Method 2: You can pull from docker hub

At the end of testing we have pushed the image to dockerhub, you can use that to run the container
the downside being that it will take some time to download the data. Use the command:
  > docker run -it --name speech mapurv/speech


to start the container

## Testing
Before testing you need to run the code in src. After that just go to the test folder and type:
  > pytest

