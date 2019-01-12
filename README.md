### 问题更新：通过 mxnet2caffe项目 转成的caffe模型 对不同图像 返回相同的结果
##### 分析: 早期版本的mobilefaceNet 项目有一个bug, 全连接层的 name为pre_fc1，但是权重名称为‘fc1_weight’，导致转换后的caffe model 对应层的权重值有问题。代码如下，bug代码为29-30行
  
```
  def get_symbol(num_classes, **kwargs):
    global bn_mom
    bn_mom = kwargs.get('bn_mom', 0.9)
    wd_mult = kwargs.get('wd_mult', 1.)
    data = mx.symbol.Variable(name="data")
    data = data - 127.5
    data = data * 0.0078125
    conv_1 = Conv(data, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1")
    conv_2_dw = Conv(conv_1, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw")
    conv_23 = DResidual(conv_2_dw, num_out=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=128, name="dconv_23")
    conv_3 = Residual(conv_23, num_block=4, num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128,
                      name="res_3")
    conv_34 = DResidual(conv_3, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=256, name="dconv_34")
    conv_4 = Residual(conv_34, num_block=6, num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256,
                      name="res_4")
    conv_45 = DResidual(conv_4, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=512, name="dconv_45")
    conv_5 = Residual(conv_45, num_block=2, num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256,
                      name="res_5")

    conv_6_sep = Conv(conv_5, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6sep")
    conv_6_dw = Linear(conv_6_sep, num_filter=512, num_group=512, kernel=(7, 7), pad=(0, 0), stride=(1, 1),
                       name="conv_6dw7_7")
    # conv_6_dw = mx.symbol.Dropout(data=conv_6_dw, p=0.4)
    
    _weight = mx.symbol.Variable("fc1_weight", shape=(num_classes, 512), lr_mult=1.0, wd_mult=wd_mult)
    conv_6_f = mx.sym.FullyConnected(data=conv_6_dw, weight=_weight, num_hidden=num_classes, name='pre_fc1')
    
    fc1 = mx.sym.BatchNorm(data=conv_6_f, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
```
###### 修复方法，将项目 mxnet2caffe下的transformer.py 的40行添加 'key_caffe = 'pre_fc1' 即可。pre_fc1是全连接层，但是 mobilefacenet在全连接层后面又添加了 batchnorm的操作，而该层的name为 'fc1',又由于 mxnet会将该层的权重命名为'fc1_weight'，偏置重命名为'fc1_bias'，很容易造成caffe的同学的误解。

###  mobilefacenet++项目转成caffe格式

1.运行json2prototxt.py 文件，将model-symbol.json 转为mobilefacenet.prototxt文件
2.修改生成的mobilefacenet.prototxt文件的第12行，将_mulscalar0 改为data
3.修改生成的mobilefacenet.prototxt文件的第1986行，将_mul1 改为bn6f
4.运行 transformer.py 文件 生成mobilefacenet.prototxt.caffemodel文件

###  mobilefacenet项目转成caffe格式

同mobilefacenet++项目项目，但是transformer.py文件需要修改为本项目的文件
