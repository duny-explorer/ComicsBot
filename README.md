<h2 align="center">

</h1> <h1 align="center">ComicsBot</h1>

<p align="center">

<img src="https://img.shields.io/badge/made%20by-dunyexplorer-blue.svg" >

<img src="https://img.shields.io/badge/python-3.10-green.svg">

<img src="https://img.shields.io/badge/aiogram-3.0.0b7-green.svg">

<img src="https://img.shields.io/badge/pytorch-2.0.1+cu117-red.svg">

<img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" >

<img src="https://img.shields.io/github/stars/duny-explorer/ComicsBot.svg?style=flat">

<img src="https://img.shields.io/github/languages/top/duny-explorer/ComicsBot.svg">

</p>

<img src="https://cubiq.ru/wp-content/uploads/2024/07/marvel-780x439.jpg" width="100%">

<h2 align="center"><a  href="https://t.me/ConventToComicsBot">Live Demo</a></h2>


## Description

This is a telegram bot that will allow anyone to try algorithms for transferring style. In our case Fast Neural Style and CycleGan

<p align="center">
  <img src="https://github.com/duny-explorer/ComicsBot/assets/37844052/a6799be6-e080-45bd-8051-f1a1fefb361a" alt="animated" />
</p>

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

## Deployment

```bash
git clone https://github.com/duny-explorer/ComicsBot
cd ComicsBot
python -m pip install --user virtualenv
python -m venv bot
source bot/bin/activate
python -m pip install -r requirements.txt
python bot/main.py
```

If there are problems with aiogram:

```bash
pip install https://github.com/aiogram/aiogram/archive/refs/heads/dev-3.x.zip
```

You can also use Dockerfile

## For the future
- [ ] Reatrian CycleGAN
- [ ] Write the CycleGAN yourself
