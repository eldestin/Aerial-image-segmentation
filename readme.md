# Aerial image segmentation repository
This is the repository of our project: Aerial image segmentation.

Current models are the reproduction based on:
1. ![XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model](https://arxiv.org/abs/2207.07115)
## Dependencies
```
pip install -r requirements.txt
```
## Download the training datasets
[UAVid Dataset](https://paperswithcode.com/dataset/uavid)

[LoveDA Dataset](https://paperswithcode.com/dataset/loveda)

## Training
```
cd train
python static_train.py -c "train_path"
or
python video_train.py
```

## Evaluation
Check the video generation.ipynb and test the video generation code;

For evaluation, modifided the train code and set the path to "val path"

## Video on school
### Static image segmentation
[![Watch the video](https://i.ytimg.com/vi/z9OjdoZb4-I/maxresdefault.jpg)](https://www.youtube.com/watch?v=z9OjdoZb4-I "")
### Video object segmentation
[![](https://i.ytimg.com/vi/8ctzlT-RHyw/maxresdefault.jpg)](https://youtu.be/8ctzlT-RHyw "")
