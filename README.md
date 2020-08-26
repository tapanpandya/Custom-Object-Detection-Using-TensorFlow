## Steps to build custom object detection in following system 
In the previous semester, I worked on a project based on custom object detection. I used following libraries/software tools to train custom model.

1. Python 3.6.8
2. TensorFlow 1.8.0 (I tried TensorFlow 2.0.0 version but had an issue with my CPU based laptop)
3. LabelImg - for creating labels within image files and generating xml for each of the images.
4. Following libraries
   	 - pip install --user opencv-python
	 - pip install --user Cython
	 - pip install --user contextlib2
	 - pip install --user pillow
	 - pip install --user lxml
	 - pip install --user jupyter
	 - pip install --user matplotlib
	Please download if I missed any.

# My system was as follow:
- Windows 10 (64 bit)
- 8-GB RAM, 256 GB SSD

model directory should have following folders

- models/object_detection/training/
- models/object_detection/traning_custom_dir/
- models/object_detection/test_train/
- models/object_detection/test_train/ssd_mobilenet_v1_coco_11_06_2017
