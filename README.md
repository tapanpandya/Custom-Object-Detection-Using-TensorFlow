## Steps to build custom object detection in following system 
In the previous semester, I worked on a project based on custom object detection. I used following libraries/software tools to train custom model.

1. Python 3.6.8
2. TensorFlow 1.8.0 (2.0.0 version tried but had an issue with my CPU based laptop)
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

Step by step process to create custom object detection model and train it.

copy files from following directories:
1)	From ->  models/object_detection/samples/ssd_mobilenet_v1_coco.config (You need to edit this file in further step)
	To -> models/object_detection/training/   (You need to create another file here called labelmap.pbtxt)

2)	From ->  models/object_detection/lagecy/train.py
	To -> models/object_detection/

## Below is the steps which I used but I might be wrong for the order in which I had done as I am writing it after 5 months or so.

Steps to develop the custom model.

Do not forgot to run following two lines(Never forget it):
After every changes that you make.
python setup.py build
python setup.py install

1) Download python 3.6.8

2) Install tensorflow-cpu using collowing command
	pip install --user tensorflow-cpu==1.15(Advanced version will give error as I already experienced.)

3) Download models-master git repository from github
	- https://github.com/tensorflow/models
	unzip the files and put it wherever you like.

4) Install some of the packages like provided below:
	pip install --user opencv-python
	pip install --user Cython
	pip install --user contextlib2
	pip install --user pillow
	pip install --user lxml
	pip install --user jupyter
	pip install --user matplotlib
	Please download if I missed any.

5) Donwload Protobuf from following link:
	https://github.com/protocolbuffers/protobuf/releases

6) Now run following command from research directory.
Go to bin folder of protoc directory(Do not change the directory)
	# From models/research/
	cd D:/Tensorflow_Files/models/research/
	D:/Tensorflow_Files/protoc/bin/protoc object_detection/protos/*.proto --python_out=.

7) Now, Just add PythonPath by following command:
	from models/research directory
	set PYTHONPATH=$PYTHONPATH:'pwd':'pwd'/slim

8) Download labelImg software to labelling the images and to save xml versions of each image
	https://tzutalin.github.io/labelImg/

	- Before using the labelImg, be sure to download atleast 200 images of
	  Chosen object(if multiple objects then for each object into 200)
	- Then divide the images in two categoties. (Make directory test and train inside test_train directory)
	your main root directory will be like:
	test_train
		test(20% of images goes here)
		train(80% of images goes here)

9) Now, generate xml files for each image with drawing boxes around the objects you want to detect.
    - Use lableImg software for labelling for the chosen objects
	
10) Next, we need to create csv file for test and train directories data using following file.
	- xml_to_csv.py

11)	Edit generate_tfrecord.py file and add class names(object names in same order as you created in labelmap.pbtxt)
	- we must change labels inside function called class_text_to_int(row_label)
		- Change row_label names to label name that you gave to your images inside labelImg software.

12) Now, just generate .record extention file for test and train directory images. By following command,
	- python generate_tfrecord.py --csv_input=image\test_train\train_labels.csv --image_dir=image\test_train\train --output_path=image\test_train\train.record
	- python generate_tfrecord.py --csv_input=image\test_train\test_labels.csv --image_dir=image\test_train\test --output_path=image\test_train\test.record

13) create a new folder and name it as training.
	- inside it copy one of the model file from object_detection/models directory.
	- I have used ssd_mobilenet_v1_coco.config file which copied from models directory.
		- modify certain fields inside ssd_mobilenet_v1_coco.config file.
			- num_classes= Number of objects you want to detect(In my case it is 3.)
			- batch_size=number of images from test directory(Only jpg files not xml). This field will be found inside train_config
			- fine_tune_checkpoint: "D:/Tensorflow_Files/models/research/object_detection/ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"
			- inside train_input_reader:
				- input_path: "D:/Tensorflow_Files/models/research/object_detection/test_train/train.record"
				- label_map_path: "D:/Tensorflow_Files/models/research/object_detection/training/labelmap.pbtxt"
			- inside eval_input_reader:
				- input_path: "D:/Tensorflow_Files/models/research/object_detection/test_train/test.record"
				- label_map_path: "D:/Tensorflow_Files/models/research/object_detection/training/labelmap.pbtxt"
	- Now, create a new file with extention labelmap.pbtxt inside training directory.

14) Now, start training of the generated .record files using following command from models/object_detection directory.
	- Before starting to train the model, do following steps.
	- copy train.py from models/object_detection/legacy folder to object_detection folder
	- then run following command.(training directory should have .config file as well as .pbtxt file.)
			python train.py --logtostderr --train_dir=training_custom_dir/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config
	
15)  Now coming to Last Step, Exporting the results after training the data, by following command. The command should be run from models/object_detection folder.
	- python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training_custom_dir/model.ckpt-35 --output_directory inference_graph 


Do not forgot to run following two lines(Never forget it):
python setup.py build
python setup.py install
  


python train.py --logtostderr --train_dir=training_custom_dir/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config
