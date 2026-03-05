
# Smart Voice - Sign Language & Speech Communication System

Smart Voice is a real-time communication system designed to bridge the gap between hearing-impaired individuals and the general public. The system recognizes hand gestures from sign language and converts them into text, while also supporting speech-to-text conversion to help enable two-way communication.

Unlike many basic gesture recognition systems that struggle with multiple hands, 
Smart Voice solves the common **two-hand detection problem** by correctly detecting 
and processing **both hands simultaneously**, enabling more accurate recognition of 
complex sign language gestures.

The project uses computer vision and machine learning to detect hand gestures through a webcam and translate them into readable text.

---


## Key Improvements

One of the common challenges in gesture recognition systems is handling multiple hands.  
Many systems fail when both hands appear in the frame.

Smart Voice addresses this limitation by implementing **two-hand landmark detection**, allowing the system to:

- Detect **both hands simultaneously**
- Extract landmarks for each hand
- Process gestures involving **two-hand coordination**
- Improve recognition accuracy for more complex signs

---
## System Architecture

#### *The system follows the pipeline below:*

1. Webcam captures hand gesture images  
2. Image preprocessing  
3. Background removal and binary conversion  
4. Feature extraction using hand landmarks  
5. Classification using an SVM model  
6. Gesture converted to text output  



#### System Design
![Data training](https://github.com/Dacrron/Smart-Voice/blob/main/SS%20for%20SmartVoice_Readme/System%20Design.png)

---
## Installation

#### 1. Clone the repository

```bash
  git clone https://github.com/Dacrron/Smart-Voice.git
```
#### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate
```
Windows (Activate it) : 
```bash
venv\Scripts\activate
```


#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run 

```bash
python main.py
```
## Screenshots

#### Accuracy 
![ACCURACY Screenshot](https://github.com/Dacrron/Smart-Voice/blob/main/ACCURACY%20IMAGE/Screenshot%202023-05-23%20231139.png)

#### Data Training Examples
![Data training](https://github.com/Dacrron/Smart-Voice/blob/main/data/4/14.jpg)

#### Login and Signup
![Login and Signup Snap](https://github.com/Dacrron/Smart-Voice/blob/main/SS%20for%20SmartVoice_Readme/Login%20and%20Signup.png)

#### UI
![UI Snap](https://github.com/Dacrron/Smart-Voice/blob/main/SS%20for%20SmartVoice_Readme/UI.png)

#### Detection
![Detection of hand snap](https://github.com/Dacrron/Smart-Voice/blob/main/SS%20for%20SmartVoice_Readme/Detection.png)


