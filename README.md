### update 11.24
# mobilefacenet项目转成caffe格式 
https://github.com/honghuCode/mobileFacenet-ncnn/tree/feature/mobilefacenet-mxnet2caffe

# 1.运行json2prototxt.py 文件，将model-symbol.json 转为mobilefacenet.prototxt文件
# 2.修改生成的mobilefacenet.prototxt文件的第12行，将_mulscalar0 改为data
# 3.修改生成的mobilefacenet.prototxt文件的第1986行，将_mul1 改为bn6f
# 4.运行 transformer.py 文件 生成mobilefacenet.prototxt.caffemodel文件

### update 5.17
### 采用insightface作者训练的mobileFaceNet 
### https://github.com/deepinsight/insightface/issues/214
#### training dataset: ms1m
## LFW: 99.50, CFP_FP: 88.94, AgeDB30: 95.91

## mobileFacenet-ncnn
## 5.16
## LFW 98.83


##### ====================================
##### 1.VS2015下 选择X64 Release模式，运行即可。opencv、ncnn依赖都已经自动配置。
##### 2.直接解压lfw-112X112.zip文件，可计算lfw数据集中同一张脸和不同脸的得分。

##### TODO ############
##### 1.将mxnet的模型转换为caffe模型 。
##### 2.将BN层和卷积层合并（提高速度）参考： https://github.com/chuanqi305/MobileNet-SSD/blob/master/merge_bn.py
##### 3.将caffe模型转换为ncnn模型。

### 增加mtcnn 做人脸对齐
https://github.com/honghuCode/mobileFacenet-ncnn/tree/update-mobilefacenet-ncnn


######## 欢迎入QQ群 707134958交流
