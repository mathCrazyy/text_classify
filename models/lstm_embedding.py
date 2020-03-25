import torch
import torch.nn as nn
import codecs
import numpy as np
import re

class Config(object):
    def __init__(self,data_ori, data_tgt):
        self.model_name="lstm_embedding"
        self.data_ori=data_ori+"/"
        self.train_path="train.csv"
        self.valid_path="valid.csv"
        self.test_path="test.csv"
        self.embedding_path="need_bertembedding"

        self.sen_max_length=150

        self.embedding_dim=768
        self.hidden_dim=128
        self.class_num=10
        self.num_lstm_layers=1
        self.num_linear=2
        self.dropout=0.3
        self.batch_size=64
        self.learning_rate=1e-3
        self.epochs = 10

        ### 构建词典
        self.vocab_maxsize = 4000
        self.vocab_minfreq = 10

        self.save_path=data_tgt+self.model_name+".ckpt"
        self.log_path=data_tgt+"/log/"+self.model_name

        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.print_cricle=100
        self.require_improvement=1000




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
        self.encoder = nn.LSTM(config.embedding_dim, config.hidden_dim, num_layers=config.num_lstm_layers, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.linear_layers = []
        for _ in range(config.num_linear):
            self.linear_layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)

        self.predictor = nn.Linear(config.hidden_dim, 10)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        feature = self.dropout(feature)
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds
