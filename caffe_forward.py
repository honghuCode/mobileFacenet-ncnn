#coding:utf-8

import caffe
import cv2
import numpy as np
from scipy.spatial.distance import pdist
import mxnet as mx
import numpy as np
from PIL import Image
from collections import namedtuple
import time
net = caffe.Net('mobilefacenet.prototxt', 'mobilefacenet.prototxt.caffemodel',caffe.TEST)
def caffeGetFeature(imgPath):
	bgr = cv2.imread(imgPath)
	#cv2.imshow("BGR",img)
	#cv2.waitKey(0)

	# BGR 0 1 2
	# RGB 2 
	rgb = bgr[...,::-1]
	rgb = (rgb - 128.0) / 128.0
	#rgb = rgb.transpose((2,1,0)) 
	rgb = np.swapaxes(rgb, 0, 2)
	rgb = np.swapaxes(rgb, 1, 2) 
	rgb = rgb[None,:] # add singleton dimension
	#cv2.imshow("RGB",rgb)
	#cv2.waitKey(0)
	#print (rgb)
	out = net.forward_all( data = rgb ) # out is probability
	#print(out['fc1'][0])
	a = out['fc1'][0]
	return a

##########################MXnet##########



 
         
#读取一张本地图片
def read_one_img(img_path):
    #这里注意是jpg，即3通道rgb，如果不是的话需要转换
    #img=Image.open(img_path)
    img = cv2.imread(img_path,1)
    #img=img.resize((112,112),Image.BILINEAR)
    #img=np.array(img)
    #return np.array([[img[:,:,i] for i in xrange(3)]])
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2) 
    Batch = namedtuple('Batch', ['data'])
    #这里也要吐槽，数据要封装成这样的形式才能用
    one_data=Batch([mx.nd.array(np.array([img]))])
    return one_data
 
def mxnetForward(img):
    #数据的shape，这样写有点复杂了，这里吐槽，哈哈
    data_shape=[('data', (1,3,112,112))]
     
    #读取一张图片，路径替换成你的图片路径
    data=read_one_img(img)
  
    #装载模型,epoch为Inception-BN-0039.params的数字
    prefix="model"
    model=mx.module.Module.load(prefix=prefix,epoch=0,context=mx.cpu())
     
    #这里bind后就可以直接用了，不需要再定义网络架构（这个比tensorflow简洁）
    model.bind(data_shapes=data_shape)
     
    #前馈过程
    #model.forward(data,is_train=False)
     
    #获取前馈过程的输出(这个设计为何？)，result类型为list([mxnet.ndarray.NDArray])
    #result = model.get_outputs()
     
    #输出numpy类型的数组
    #result=result[0].asnumpy()
     
    #下面取最大值作为预测值 
    #pos=np.argmax(result)
    ##print 'max:',np.max(result)
    #print 'position:',pos
    #print "result:",clazz_map[pos]
     
     
    #获取网络结构
    internals=model.symbol.get_internals()
    #print '\n'.join(internals.list_outputs())
     
    #获取从输入到flatten层的特征值层的模型
    feature_net = internals["fc1_output"]
    feature_model=mx.module.Module(symbol=feature_net,context=mx.cpu())
    feature_model.bind(data_shapes=data_shape)
     
    #获取模型参数
    arg_params,aux_params=model.get_params()
     
    #上面只是定义了结构，这里设置特征模型的参数为Inception_BN的
    feature_model.set_params(arg_params=arg_params,
                             aux_params=aux_params, 
                             allow_missing=True)
     
    #输出特征
    feature_model.forward(data,is_train=False)
    feature = feature_model.get_outputs()[0].asnumpy()
     
    #print 'shape:',np.shape(feature)
    #print 'feature:',feature
    return feature
#a = caffeGetFeature("Peter_Struck_0005.jpg")
#print (a)
#b = test()[0]
#print(a)
#print(b)
#a = mxnetForward("Peter_Struck_0005.jpg")
#b = mxnetForward("Peter_Struck_0003.jpg")
#a = caffeGetFeature("Peter_Struck_0001.jpg")
#b = caffeGetFeature("Zhang_Wenkang_0001.jpg")
#d1=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
#d2 = 1 - pdist(np.vstack([a,b]),'cosine')
#print(d1,d2)

counter = 0
t1 = time.time()
rs = []
dir = "lfwAligned/"
with open('pairs_1.txt') as f:
	lines = f.readlines()
	#print(len(lines))
	for line in lines:
		line = line.strip()
		arr = line.split(",")
		Limg = dir + arr[0]
		Rimg = dir + arr[1]
		#print Limg,Rimg
		gtoundTruthLable = arr[2]
		Lfeat = caffeGetFeature(Limg)
		Rfeat = caffeGetFeature(Rimg)
		#print(arr[2])
		cosSim =  1 - pdist(np.vstack([Lfeat,Rfeat]),'cosine')
		#print(gtoundTruthLable,"===>>",type(cosSim),cosSim[0])
		rs.append(gtoundTruthLable + "," + str(cosSim[0]))
		counter = counter + 1
		t2 = time.time()
		if counter % 100 ==0:
			print(counter,t2 - t1)
			t1 = t2
			#break
		#print(line)


with open('rs.txt','w') as f:
	for r in rs:
		f.write(r + "\n")







