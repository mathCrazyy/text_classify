from torchtext.data import Field
from torchtext.data import Iterator,BucketIterator
from torchtext.vocab import Vectors

import torch
from torchtext.data import TabularDataset

def generate_data(config):
    ## 不同字段的操作定义

    tokenizer = lambda x: [one for one in x]
    TEXT = Field(sequential=True, tokenize=tokenizer,fix_length=config.sen_max_length)##截断句长直接影响acc!!!
    LABEL = Field(sequential=False, use_vocab=False)  ## 如果标签是数值型的话

    datafields = [("context", TEXT), ("label_id", LABEL)]  ## TEXT field, LABEL field
    test_field = [("context", TEXT), ("label_id", LABEL)]
    train_file, valid_file = TabularDataset.splits(
        path=config.data_ori,
        train=config.train_path,
        validation=config.valid_path,
        format="csv",
        skip_header=True,
        fields=datafields
    )
    test_file = TabularDataset(
        path=config.data_ori+config.test_path,
        format="csv",
        skip_header=True,
        fields=test_field
    )
    ## 构建词典
    vectors=Vectors(name=config.data_ori+config.embedding_path,cache="./")
    TEXT.build_vocab(train_file,max_size=config.vocab_maxsize, min_freq=config.vocab_minfreq, vectors=vectors)
    TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)

    train_iter, val_iter = BucketIterator.splits(
        (train_file, valid_file),
        batch_sizes=(config.batch_size, config.batch_size),
        device=config.device,
        sort_key=lambda x: len(x.context),
        sort_within_batch=True,
        # 当要使用pack_padded_sequence时，需要将sort_within_batch设置为True，同时会将paded sequence 转为PackedSequence对象
        repeat=False
    )

    test_iter = Iterator(test_file, batch_size=config.batch_size, device=config.device, sort=False, sort_within_batch=False, repeat=False)

    return train_iter, val_iter, test_iter, TEXT

if __name__=="__main__":
    print("test data")
    #train_iter, valid_iter, test_iter=generate_data(file_path)

    #a=list(train_iter)
    #print(a[0])
    #print(a[0].context)
