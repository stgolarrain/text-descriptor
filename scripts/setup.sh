#!/bin/bash

mkdir dataset
wget -O dataset/vqa2.zip http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip dataset/vqa2.zip -d dataset/vqa2
rm dataset/vqa2.zip
