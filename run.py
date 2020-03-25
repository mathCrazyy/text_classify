import time
import torch
import numpy as np
from importlib import import_module
from utils import generate_data
from train_eval import train, test


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
   
    ## 测试单条句子
    sentence="哈哈哈哈你好啊啊"
    sentence="""鲍勃库西奖归谁属？ NCAA最强控卫是坎巴还是弗神新浪体育讯如今，本赛季的NCAA进入到了末段，各项奖项的评选结果也即将出炉，
其中评选最佳控卫的鲍勃-库西奖就将在下周最终四强战时公布，鲍勃-库西奖是由奈史密斯篮球名人堂提供，旨在奖励年度最佳大学控
卫。最终获奖的球员也即将在以下几名热门人选中产生。〈〈〈 NCAA疯狂三月专题主页上线，点击链接查看精彩内容吉梅尔-弗雷戴特
，杨百翰大学“弗神”吉梅尔-弗雷戴特一直都备受关注，他不仅仅是一名射手，他会用“终结对手脚踝”一样的变向过掉面前的防守>者，并且他可以用任意一支手完成得分，如果他被犯规了，可以提前把这两份划入他的帐下了，因为他是一名命中率高达90%的罚球手>。弗雷戴特具有所有伟大控卫都具备的一点特质，他是一位赢家也是一位领导者。“他整个赛季至始至终的稳定领导着球队前进，这是
无可比拟的。”杨百翰大学主教练戴夫-罗斯称赞道，“他的得分能力毋庸置疑，但是我认为他带领球队获胜的能力才是他最重要的控>卫职责。我们在主场之外的比赛(客场或中立场)共取胜19场，他都表现的很棒。”弗雷戴特能否在NBA取得成功？当然，但是有很多专>业人士比我们更有资格去做出这样的判断。“我喜爱他。”凯尔特人主教练多克-里弗斯说道，“他很棒，我看过ESPN的片段剪辑，从>剪辑来看，他是个超级巨星，我认为他很成为一名优秀的NBA球员。”诺兰-史密斯，杜克大学当赛季初，球队宣布大一天才控卫凯瑞->厄尔文因脚趾的伤病缺席赛季大部分比赛后，诺兰-史密斯便开始接管球权，他在进攻端上足发条，在ACC联盟(杜克大学所在分区)的得
分榜上名列前茅，但同时他在分区助攻榜上也占据头名，这在众强林立的ACC联盟前无古人。“我不认为全美有其他的球员能在凯瑞-厄
尔文受伤后，如此好的接管球队，并且之前毫无准备。”杜克主教练迈克-沙舍夫斯基赞扬道，“他会将比赛带入自己的节奏，得分，>组织，领导球队，无所不能。而且他现在是攻防俱佳，对持球人的防守很有提高。总之他拥有了辉煌的赛季。”坎巴-沃克，康涅狄格>大学坎巴-沃克带领康涅狄格在赛季初的毛伊岛邀请赛一路力克密歇根州大和肯塔基等队夺冠，他场均30分4助攻得到最佳球员。在大东
赛区锦标赛和全国锦标赛中，他场均27.1分，6.1个篮板，5.1次助攻，依旧如此给力。他以疯狂的表现开始这个赛季，也将以疯狂的表
现结束这个赛季。“我们在全国锦标赛中前进着，并且之前曾经5天连赢5场，赢得了大东赛区锦标赛的冠军，这些都归功于坎巴-沃克>。”康涅狄格大学主教练吉姆-卡洪称赞道，“他是一名纯正的控卫而且能为我们得分，他有过单场42分，有过单场17助攻，也有过单>场15篮板。这些都是一名6英尺175镑的球员所完成的啊！我们有很多好球员，但他才是最好的领导者，为球队所做的贡献也是最大。”
乔丹-泰勒，威斯康辛大学全美没有一个持球者能像乔丹-泰勒一样很少失误，他4.26的助攻失误在全美遥遥领先，在大十赛区的比赛中
，他平均35.8分钟才会有一次失误。他还是名很出色的得分手，全场砍下39分击败印第安纳大学的比赛就是最好的证明，其中下半场他
曾经连拿18分。“那个夜晚他证明自己值得首轮顺位。”当时的见证者印第安纳大学主教练汤姆-克雷恩说道。“对一名控卫的所有要>求不过是领导球队、使球队变的更好、带领球队成功，乔丹-泰勒全做到了。”威斯康辛教练博-莱恩说道"""
    print(sentence)
    res= test(config, model,TEXT,sentence)
    print(res)
    sentence="""景顺长城就参与新股询价问题作出说明⊙本报记者 黄金滔 安仲文 景顺长城基金管理有限公司11日在其网站上就参与新股询价有关问
题作出说明，表示该公司本着独立、客观、诚信的原则参与新股询价，遵守相关法律法规和公司内部制度，并切实保障了基金持有人利
益，并表示将严肃对待中国证券业协会对其提出的自律处理。景顺长城表示，作为中国证券业协会认定的IPO询价对象，该公司认真履>行询价义务，2008年度共参与7只新股询价。根据该公司报价区间和实际发行价格的比较，不存在较大价格偏离现象及操纵价格嫌疑。>对于公司参与询价的股票，除其中1只股票由于公司旗下基金参与了同一发行期内另外5只股票网上申购而没有申购外，其余6只股票全>部参与网上申购。据了解，根据《关于基金投资非公开发行股票等流通受限证券有关问题的通知》（证监基金字[2006]141号），为了>切实保护基金持有人利益、防范基金资产的流动性风险，景顺长城公司于2006年7月修改投资管理制度，明确规定“不投资带有锁定期>的证券”，因此，从2006年8月起，该公司没有投资任何带有锁定期的股票，包括IPO网下申购；并且在每次参与新股询价时，公司在递
交的《新股询价信息表》中的“在报价区间的申购意向”一栏均明确填写“不确定”或“不申购”。不过，景顺长城也强调，由于中国
证券市场处于快速发展时期，法规制度不断完善，该公司将严肃对待中国证券业协会对其提出的自律处理，一如既往地以审慎诚信的态
度为投资人服务"""
    res=test(config, model , TEXT, sentence)
    print(res)







