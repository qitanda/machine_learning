# -*- coding: utf-8 -*-
# @Time : 2023/3/16 16:42
# @Author : Jclian91
# @File : model_predict.py
# @Place : Minghang, Shanghai
import torch as T
import numpy as np
import torch.nn.functional as F

# from model import TextClassifier
from text_featuring import load_file_file, text_feature

print("load model")
model = T.load('sougou_mini_cls.pth')

label_dict, char_dict = load_file_file()
label_dict_rev = {v: k for k, v in label_dict.items()}
# print(label_dict)
# print(char_dict)
text = '盖世汽车讯，特斯拉去年击败了宝马，夺得了美国豪华汽车市场的桂冠，并在今年实现了开门红。1月份，得益于大幅降价和7500美元美国电动汽车税收抵免，特斯拉再度击败宝马，蝉联了美国豪华车销冠，并且注册量超过了排名第三的梅赛德斯-奔驰和排名第四的雷克萨斯的总和。根据Experian的数据，在所有豪华品牌中，1月份，特斯拉在美国的豪华车注册量为49，917辆，同比增长34%；宝马的注册量为31，070辆，同比增长2.5%；奔驰的注册量为23，345辆，同比增长7.3%；雷克萨斯的注册量为23，082辆，同比下降6.6%。奥迪以19，113辆的注册量排名第五，同比增长38%。凯迪拉克注册量为13，220辆，较去年同期增长36%，排名第六。排名第七的讴歌的注册量为10，833辆，同比增长32%。沃尔沃汽车排名第八，注册量为8，864辆，同比增长1.8%。路虎以7，003辆的注册量排名第九，林肯以6，964辆的注册量排名第十。'
text = "北京时间3月16日，NBA官方公布了对于灰熊球星贾-莫兰特直播中持枪事件的调查结果灰熊，由于无法确定枪支是否为莫兰特所有，也无法证明他曾持枪到过NBA场馆，因为对他处以禁赛八场的处罚，且此前已禁赛场次将算在禁赛八场的场次内，他最早将在下周复出。"
text = "3月11日，由新浪教育、微博教育、择校行联合主办的“新浪&微博2023国际教育春季巡展•深圳站”于深圳凯宾斯基酒店成功举办。深圳优质学校亮相展会，上千组家庭前来逛展。近30所全国及深圳民办国际化学校、外籍人员子女学校、公办学校国际部等多元化、多类型优质学校参与了本次活动。此外，近10位国际化学校校长分享了学校的办学特色、教育理念及学生的成长案例，参展家庭纷纷表示受益匪浅。展会搭建家校沟通桥梁，帮助家长们合理规划孩子的国际教育之路。深圳国际预科书院/招生办主任沈兰Nancy Shen参加了本次活动并带来了精彩的演讲，以下为演讲实录："
text = "据中国、伊朗、俄罗斯等国军队共识，3月15日至19日，中、伊、俄等国海军将在阿曼湾举行“安全纽带-2023”海上联合军事演习。“安全纽带-2023”海上联演由2019年、2022年先后两次举行的中伊俄海上联演发展而来，中方派出南宁号导弹驱逐舰参演，主要参加空中搜索、海上救援、海上分列式等科目演练。此次演习有助于深化参演国海军务实合作，进一步展示共同维护海上安全、积极构建海洋命运共同体的意愿与能力，为地区和平稳定注入正能量。"
text = "指导专家：皮肤科教研室副主任、武汉协和医院皮肤性病科主任医师冯爱平教授在临床上，经常能看到有些人出现反复发作的口腔溃疡，四季不断，深受其扰。其实这已不单单是口腔问题，而是全身疾病的体现，特别是一些免疫系统疾病，不仅表现在皮肤还会损害黏膜，下列几种情况是造成“复发性口腔溃疡”的原因。缺乏维生素及微量元素。缺乏微量元素锌、铁、叶酸、维生素B12等时，会引发口角炎。很多日常生活行为可能造成维生素的缺乏，如过分淘洗米、长期进食精米面、吃素食等，很容易造成B族维生素的缺失。"

labels, contents = ['汽车'], [text]
samples, y_true = text_feature(labels, contents, label_dict, char_dict)
print(samples)
print(len(samples[0]))
x = T.from_numpy(np.array(samples)).long()
y_pred = model(x)
print(y_pred)
y_numpy = F.softmax(y_pred, dim=1).detach().numpy()
print(y_numpy)
predict_list = np.argmax(y_numpy, axis=1).tolist()
for i, predict in enumerate(predict_list):
    print(f"第{i+1}个文本，预测标签为： {label_dict_rev[predict]}")

