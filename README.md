Speech Recognition
===================

This project's objective is  to recognize speech without triggers.
Trigger means a command indicating the start of an instruction such as "OK google" or "Hey siri".


## Description

Follow the steps below to speech recognition.
1. Record instruction voice
2. Convert recorded audio data to mfcc (Mel-Frequency Cepstrum Coefficients)
3. Infer the mfcc with CNN

The registered voice commands are as follows. (Can be added other commands)
 - 「電気をつけて」(Turn on the light)
 - 「電気を消して」(Turn off the light)
 - 「エアコンつけて」(Turn on the air conditioner)
 - 「エアコン消して」(Turn off the air conditioner)
 - 「カーテン開けて」(Open the curtain)
 - 「カーテン閉めて」(Close the curtain)
 - 「テレビをつけて」(Turn on the TV)
 - 「テレビを消して」(Turn off the TV)
 - 「扇風機をつけて」(Turn on the fan)
 - 「扇風機を消して」(Turn off the fan)

## Requirement

- Python 3.5.2
- Tensorflow 2.0.0a0
- Keras 2.2.4


## Installation

```
$ git clone https://github.com/Crtv-info/Speech_Reicognition
$ cd Speech_Recognition
```

## Execution
1. Run a program to record audio
```
$ python3 detection.py
```
2. Run a program to infer
```
$ python3 api.py
```