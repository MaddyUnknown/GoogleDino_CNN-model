# GoogleDino_CNN-model
This model used CNN to Play the Google Dino game.

![application example](https://github.com/MaddyUnknown/GoogleDino_CNN-model/blob/master/Test/Image/ezgif.com-crop.gif)

_______________________
###### Important:
- The google dino screen must be keep on the left side of the screen since the program grabs the screen to feed as input to CNN model
- The screen grab is adjusted for resolution 1920x1080 users with other resolution settings might need to adjust the window for screen grab

![Placement example](https://github.com/MaddyUnknown/GoogleDino_CNN-model/blob/master/Test/Image/allignment2.png)
_______________________


## How the Application works:
1. Takes a screenshot of the browser
2. Scales it down to model input size (140,40) and convert to gray-scale
3. Concatenate 3 such screenshort
4. Predict the decision using trained CNN (0 for up, 1 for hold, 2 for down) 
5. Gives input to the browser using pyautogui

![3-channel example](https://github.com/MaddyUnknown/GoogleDino_CNN-model/blob/master/Test/Image/3%20channel.PNG)

> ###### 3 channels consisting of 3 consecutive frames to predict a single output

## Repo Contents:
- `ModelRun.py` - Currently the main program file, this is responsible for all the steps described in how the application works, currently uses model 9 in the model folder
- `Model` - Contain all the model trained (All models dont use the same architecture used in ModelRun.py
- `CreateData.py` - Helps in recording training data
- `Sequencing_data.py` - Takes the recording data and concatenates 3 images (in sequence) to give a 3x (img_dim) input for the model
- `Create_train_val.py` - Uses the recorded data to create training and validation data
- `Model_Train.ipynb` - Jupyter notebook for training model
- `Test` - Contain ModelRun for previously trained models

## Requirement:
- `PyTorch GPU v1.0.0`
- `Numpy v1.17.3`
- `cv2 v4.1.1`
- `PyAutoGUI v0.9.50`
