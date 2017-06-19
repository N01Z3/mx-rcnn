#!/usr/bin/env bash

PYTHON_BIN=/usr/bin/python

cp ~/kaggle/noaa_sea_lions/data/Train_0.txt ~/dataset/sea_lions/Train.txt
$PYTHON_BIN train_noaa.py
mkdir /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold0
mv model/log /media/n01z3/wd_storage/dataset/noaa_sea_lines/log_fold0
mv model/vgg-* /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold0

cp ~/kaggle/noaa_sea_lions/data/Train_1.txt ~/dataset/sea_lions/Train.txt
$PYTHON_BIN train_noaa.py
mkdir /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold1
mv model/log /media/n01z3/wd_storage/dataset/noaa_sea_lines/log_fold1
mv model/vgg-* /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold1

cp ~/kaggle/noaa_sea_lions/data/Train_2.txt ~/dataset/sea_lions/Train.txt
$PYTHON_BIN train_noaa.py
mkdir /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold2
mv model/log /media/n01z3/wd_storage/dataset/noaa_sea_lines/log_fold2
mv model/vgg-* /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold2

cp ~/kaggle/noaa_sea_lions/data/Train_3.txt ~/dataset/sea_lions/Train.txt
$PYTHON_BIN train_noaa.py
mkdir /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold3
mv model/log /media/n01z3/wd_storage/dataset/noaa_sea_lines/log_fold3
mv model/vgg-* /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold3

cp ~/kaggle/noaa_sea_lions/data/Train_4.txt ~/dataset/sea_lions/Train.txt
$PYTHON_BIN train_noaa.py
mkdir /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold4
mv model/log /media/n01z3/wd_storage/dataset/noaa_sea_lines/log_fold4
mv model/vgg-* /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold4

cp ~/kaggle/noaa_sea_lions/data/Train_5.txt ~/dataset/sea_lions/Train.txt
$PYTHON_BIN train_noaa.py
mkdir /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold5
mv model/log /media/n01z3/wd_storage/dataset/noaa_sea_lines/log_fold5
mv model/vgg-* /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold5

cp ~/kaggle/noaa_sea_lions/data/Train_6.txt ~/dataset/sea_lions/Train.txt
$PYTHON_BIN train_noaa.py
mkdir /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold6
mv model/log /media/n01z3/wd_storage/dataset/noaa_sea_lines/log_fold6
mv model/vgg-* /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold6

cp ~/kaggle/noaa_sea_lions/data/Train_7.txt ~/dataset/sea_lions/Train.txt
$PYTHON_BIN train_noaa.py
mkdir /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold7
mv model/log /media/n01z3/wd_storage/dataset/noaa_sea_lines/log_fold7
mv model/vgg-* /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold7

cp ~/kaggle/noaa_sea_lions/data/Train_8.txt ~/dataset/sea_lions/Train.txt
$PYTHON_BIN train_noaa.py
mkdir /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold8
mv model/log /media/n01z3/wd_storage/dataset/noaa_sea_lines/log_fold8
mv model/vgg-* /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold8

cp ~/kaggle/noaa_sea_lions/data/Train_9.txt ~/dataset/sea_lions/Train.txt
$PYTHON_BIN train_noaa.py
mkdir /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold9
mv model/log /media/n01z3/wd_storage/dataset/noaa_sea_lines/log_fold9
mv model/vgg-* /media/n01z3/wd_storage/dataset/noaa_sea_lines/fold9