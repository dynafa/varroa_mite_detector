#!/usr/bin/env bash

cd ~/Workspaces/Workspace_1/EIT2019/ML/Testing/Mask_RCNN/classes/
ls

rm drone/images/*
rm worker/images/*
rm varroa/images/*
rm queen/images/*


ls -l drone/images/
ls -l worker/images/
ls -l queen/images/
ls -l varroa/images/

cp drone\ bee\ images/* drone/images/
ls drone/images

cp queen\ bee\ images/* queen/images/
ls queen/images/

cp varroa\ mite\ on\ bee/* varroa/images/
ls varroa/images

cp worker\ bee\ images/* worker/images
ls worker/images
