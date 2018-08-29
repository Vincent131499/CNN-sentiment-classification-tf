CNN-sentiment-classification-tf：采用tensorflow工具搭建神经网络对文本进行情感分类判别
==================================================================================
网络模型为：
-----------
embedding layer -> conv layer -> max-pooling layer -> dropout layer -> softmax layer<br>

环境：
-----
Python 3.6<br>
Tensorflow 1.8<br>
numpy<br>

项目架构：
---------
1）data文件夹：存放数据集文本；<br>
2）CNN_model_with_L2.py：搭建的CNN模型，采用了L2正则化避免过拟合<br>
3）CNN_model_without_L2.py：搭建的CNN模型，没有采用L2正则<br>
4）eval.py：对给定的样本进行评估<br>
5）model_train_with_L2.py：针对采用L2正则的CNN模型进行训练，并在训练中输出loss和精度，写入到指定文件夹<br>
6）model_train_without_L2.py：针对未采用L2正则的CNN模型进行训练，并在训练中输出loss和精度，写入到指定文件夹<br>
7）process_data.py：数据预处理文件，读取数据，进行数据清洗，生成batch<br>
 
项目运行：
---------
  python model_train_without_L2.py（此处应该保证数据集路径正确：/data/rt-polaritydata/）<br>

注意：
----
该项目是对论文[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)的简单实现
