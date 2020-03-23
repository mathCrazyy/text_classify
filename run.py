import time
import torch
import numpy as np
from importlib import import_module
from utils import generate_data
from train_eval import train


import argparse

parser=argparse.ArgumentParser(description="文本分类")
parser.add_argument("--model", type=str,required=True, help="choose a model lstm")
parser.add_argument("--embedding",default="pre_trained",type=str,help="random or pre_trained")
parser.add_argument("--data_path",type=str,default="data/",help="all pred_files")
parser.add_argument("--target_path",type=str,default="data_tgt/",help="all files generated")



args=parser.parse_args()

print(args)

if __name__=="__main__":
    model_name=args.model
    data_path=args.data_path
    target_path=args.target_path

    which_model=import_module("models."+model_name)
    config=which_model.Config(data_path,target_path)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministc=True

    start_time=time.time()
    train_iter, valid_iter, test_iter, TEXT=generate_data(config)
    end_time=time.time()
    print("time usage: ",end_time-start_time)
    model=which_model.Model(config).to(config.device)
    train(config,model,train_iter,valid_iter, test_iter)






