# mobilefacenet项目转成caffe格式

1.运行json2prototxt.py 文件，将model-symbol.json 转为mobilefacenet.prototxt文件
2.修改生成的mobilefacenet.prototxt文件的第12行，将_mulscalar0 改为data
3.修改生成的mobilefacenet.prototxt文件的第1986行，将_mul1 改为bn6f
4.运行 transformer.py 文件 生成mobilefacenet.prototxt.caffemodel文件
