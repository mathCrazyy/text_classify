from data import *

#from model_embedding import Model
#from models.lstm_embedding import Model

import torch
import tqdm
from torchtext.data import Iterator, BucketIterator
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn import metrics

import numpy as np
from tensorboardX import SummaryWriter
import time

def evaluate(config, model, eval_iter, test=False):
    model.eval()
    val_loss = 0.0
    all_acc = 0.0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_total=0.0
    with torch.no_grad():
        for batch in eval_iter:
            preds = model(batch.context)
            loss=F.cross_entropy(preds.cpu(),batch.label_id.cpu())
            loss_total+=loss.item()
            predic=torch.max(preds.data,1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
            labels_all = np.append(labels_all, batch.label_id.cpu())

        acc=metrics.accuracy_score(labels_all,predict_all)
        # class_report=classification_report(predict_all,labels_all)
        class_report = classification_report(labels_all, predict_all)
        # print(labels_all)
        # print(predict_all)
    return loss_total/len(eval_iter),acc, class_report


def test(config, model, TEXT, sentence):
    sentence_seq=[TEXT.vocab.stoi[one] for one in sentence]
    need_pad=config.sen_max_length-len(sentence_seq)
    for _ in range(need_pad):
        sentence_seq.append(1)

    example=torch.Tensor(sentence_seq).long().to(config.device)
    example=example.unsqueeze(1)

    preds=model(example)
    predic=torch.max(preds.data,1)[1].cpu().numpy()
    return predic



def train(config, model, train_iter, valid_iter, test_iter):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    eval_best_loss = float("inf")
    epochs = config.epochs
    writer=SummaryWriter(log_dir=config.log_path+"/"+time.strftime("%m-%d_%H.%M",time.localtime()))
    total_batch=0
    last_improve=0
    flag=False

    for epoch in range(1, epochs + 1):
        # if epoch % 5 ==0:
        #    for p in optimizer.param_groups:
        #        p["lr"]*=0.9
        running_loss = 0.0
        runing_corrects = 0
        model.train()

        for batch in train_iter:
            total_batch+=1
            model.zero_grad()
            preds = model(batch.context)
            ## 应当是10行，5列的样子
            # y_p = batch.label_id.squeeze(1)
            y_p = batch.label_id
            # https://blog.csdn.net/ccbrid/article/details/90610599
            loss = F.cross_entropy(preds, y_p.long()).to(config.device)

            loss.backward()
            optimizer.step()
            if(total_batch%100==0):
                pred_res = torch.max(preds.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(y_p.cpu(), pred_res)
                eval_loss, eval_acc, eval_report = evaluate(config, model, valid_iter)
                test_loss, test_acc, test_report = evaluate(config, model, test_iter)
                print("train_loss: ",loss,"train_acc: ",train_acc,total_batch)
                print("eval_loss: ",eval_loss,"eval_acc: ",eval_acc,total_batch)
                print("test_loss: ",test_loss,"test_acc: ",test_acc,total_batch)
                if eval_loss < eval_best_loss:
                    eval_best_loss = eval_loss
                    torch.save(model.state_dict(), config.save_path)
                    last_improve=total_batch

                writer.add_scalar("loss/train",loss.item(),total_batch)
                writer.add_scalar("loss/dev",eval_loss,total_batch)
                writer.add_scalar("acc/train",train_acc,total_batch)
                writer.add_scalar("acc/dev",eval_acc,total_batch)
                model.train()
            if total_batch-last_improve>config.require_improvement and last_improve!=0:
                print(total_batch-last_improve)
                print(config.require_improvement)
                print("超过",config.require_improvement,"轮次没有提升并退出")
                print("eval_report: ", eval_report)
                print("test_report", test_report)
                flag=True
                break
        if flag:
            break





"""

config = Config()

nh = 64
model = SimpleLSTMBaseline(TEXT, nh).to(device)

train(config, model, train_iter, valid_iter, test_iter)


"""


