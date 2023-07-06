<h1 align="center">ComicsBot</h1>
<h2 align="center">

</h2>

<p align="center">

<img src="https://img.shields.io/badge/made%20by-dunyexplorer-blue.svg" >

<img src="https://img.shields.io/badge/python-3.10-green.svg">

<img src="https://img.shields.io/badge/aiogram-3.0.0b7-green.svg">

<img src="https://img.shields.io/badge/pytorch-2.0.1+cu117-red.svg">

<img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" >

<img src="https://img.shields.io/github/stars/duny-explorer/ComicsBot.svg?style=flat">

<img src="https://img.shields.io/github/languages/top/duny-explorer/ComicsBot.svg">

</p>

<img src="https://www.megacitycomics.co.uk/acatalog/avengers-banner.jpg" width="100%">

<h2 align="center"><a  href="https://t.me/ConventToComicsBot">Live Demo</a></h2>


## Description

This is a telegram bot that will allow anyone to try algorithms for transferring style. In our case Fast Neural Style and CycleGan

## Models

- Fast Neural Style

<img src="https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/RTNS.png" align="center">

The model and its training are presented in the file FastCNN.ipynb

- CycleGAN

<img src="https://images.deepai.org/converted-papers/1905.02200/workflow_cyclegan.jpg" align="center">

The model was taken from <a href='https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix'>this</a> repository

## Data

The following datasets were used for training:
- <a href='https://cocodataset.org/#home'>COCO 2014 dataset</a>
- <a href='https://www.kaggle.com/datasets/cenkbircanoglu/comic-books-classification'>Comic Books Images</a>
- <a href='https://github.com/Sxela/face2comics?ysclid=ljr2nd764h16820479'>face2comics by Sxela (Alex Spirin)</a>

## Project setup

```
git clone https://github.com/duny-explorer/ComicsBot
cd ComicsBot
python -m pip install --user virtualenv
python -m venv bot
python -m pip install -r requirements.txt
python bot main.py
```

## Future scope

- Write the CycleGAN yourself and retrain it
