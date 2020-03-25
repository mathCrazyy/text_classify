# text_classify
针对Cnews数据集进行分类，使用了torchtext进行文本预处理，使用textcnn,lstm提取特征做分类。  
代码中使用的need_bertembedding可以通过友军自动生成: https://github.com/xmxoxo/BERT-Vector

### 数据集如下
链接：https://pan.baidu.com/s/1t-MGwuntLgjOwlJKHh3oNg 
提取码：j2yr

## 代码主体
- models
  网络定义，包含了textcnn和lstm的网络构建。
- train_eval.py
  训练代码，验证代码，测试单条数据代码，可直接微小改动构建flask服务。
- utils.py
  数据处理部分，主要使用torchtext完成了数据的词典映射，pad，shuffle等操作。
- data
  data文件夹从百度网盘下载，直接考入即可。
- data_tag
  过程中生成的文件，包括模型和日志。
## 结果
结果具有随机性，大致差不多如下: 
![avatar](https://github.com/mathCrazyy/text_classify/tree/master/pic/eval.png)

![avatar](https://github.com/mathCrazyy/text_classify/tree/master/pic/test.png)

### reference
https://github.com/649453932/Chinese-Text-Classification-Pytorch  
http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
