# tensorflow(tf) cpp training interface for segmentation and detection 

This is the C++ training and teseting interface of tensorflow on windows env. 
tf C++ lib is needed (<https://www.tensorflow.org/install/lang_c>)

# Arch
- I use python to defind the network (U-net)
- using C++ internface to train the network
- using OpenCV to finish post processing (water shed and image denoise) 

#How to use
1.  download tf c++ lib (<https://www.tensorflow.org/install/lang_c>)
2.  install opencv and python
3.  adding lib into your visual studio env.
4.  building a visual studio project.
5.  running it!
