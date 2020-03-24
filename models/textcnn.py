import torch
import torch.nn as nn
import codecs
import numpy as np
import re
import torch.nn.functional as F

class Config(object):
    def __init__(self,data_ori, data_tgt):
        self.model_name="textcnn"
        self.data_ori=data_ori+"/"
        self.train_path="train_100.csv"
        self.valid_path="valid_100.csv"
        self.test_path="test_100.csv"
        self.embedding_path="need_bertembedding"

        self.sen_max_length=150

        self.embedding_dim=768
        self.hidden_dim=128
        self.class_num=10
        #self.num_lstm_layers=1
        #self.num_linear=2
        self.dropout=0.3
        self.batch_size=64
        self.learning_rate=1e-3
        self.epochs = 10

        self.filter_sizes=(2,3,4)
        self.num_filters=256## channels数目

        ### 构建词典
        self.vocab_maxsize = 4000
        self.vocab_minfreq = 10

        self.save_path=data_tgt+self.model_name+".ckpt"
        self.log_path=data_tgt+"/log/"+self.model_name

        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.print_cricle=100
        self.require_improvement=100

class Model(nn.Module):
    #def __init__(self, n_vocab, hidden_dim, emb_dim=100, num_linear=1):
    def __init__(self, config):
        super().__init__()
        lines = codecs.open(config.data_ori+config.embedding_path, encoding="utf-8")
        #pattern = re.compile("[\u4e00-\u9fa5a-zA-Z0-9]")
        embeddings_vec = [line.replace("\n", "") for line in lines][1:-1]

        embeddings = np.random.rand(len(embeddings_vec), config.embedding_dim)
        for index, line in enumerate(embeddings_vec):
            line_seg = line.split(" ")
            try:
                embeddings[index] = [float(one) for one in line_seg[1:]]
            except:
                # print(embeddings[index])
                pass

        pretrained_weight = np.array(embeddings)
        embeds = nn.Embedding(len(embeddings), config.embedding_dim)
        embeds.weight.data.copy_(torch.from_numpy(pretrained_weight))

        # self.embedding=nn.Embedding(n_vocab,emb_dim)
        self.embedding = embeds
        self.conv=nn.Conv2d(1,config.num_filters,(3,config.embedding_dim))
        self.convs=nn.ModuleList(
            [nn.Conv2d(1,config.num_filters,(k,config.embedding_dim))for k in config.filter_sizes]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc=nn.Linear(config.num_filters*len(config.filter_sizes),config.class_num)  # 每个核心的大小，出一个向量


    def conv_and_pool(self,inputs, conv):
        inputs=F.relu(conv(inputs))## 这里卷积后，会把embeddings所在那一层抹掉
        inputs=inputs.squeeze(3)
        inputs=F.max_pool1d(inputs,inputs.size(2))## 这里最大池化层后，会把  抹掉
        inputs=inputs.squeeze(2)
        #print(inputs.shape)
        return inputs

    def forward(self, seq):
        ## 这里很方，觉得大致思路就是，从输入到输出，最后在接fc层的时候，怼成一个[batch_size, n]大小的矩阵就可以了。n是描述特征的
        #print(seq.shape)
        seq_embedings = self.embedding(seq.t())
        #print("加了embedding然后batch放到第一维度", seq_embedings.shape)
        seq_embedings_batch=seq_embedings.unsqueeze(1)
        #print("再加一个维度",seq_embedings_batch.shape)
        ##把所有核心的结果连在一起
        xx=self.conv_and_pool(seq_embedings_batch,self.conv)
        #print(xx.shape)
        concat_res=torch.cat([self.conv_and_pool(seq_embedings_batch,conv)for conv in self.convs],1)## 注意concat的方向
        #print("所有结果链接在一起：", concat_res.shape)
        out=self.dropout(concat_res)
        out=self.fc(out)
        #print("outshape: ",out.shape)
        return out

