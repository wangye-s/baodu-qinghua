import json
import random

import numpy as np
import torch
import torch.nn as nn

'''
完成一个NLP的简易中文分类任务：当字符串中出现中文时，为正样本，其余为负样本
'''
#构建模型
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        #embedding层
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #(字符集中字符的总数，每个字符向量化后的向量维度)
        #池化层   ()
        self.pool = nn.AvgPool1d(sentence_length)  #将sentence_length维 池化成 1维
        #rnn
        self.rnn_classifier = nn.RNN(vector_dim, 2, batch_first=True)
        # 线性层
        # self.liner = nn.Linear(2, 1)
        #激活函数
        self.activate = torch.sigmoid
        #损失函数
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)  #交叉熵
        # self.loss = nn.functional.mse_loss
    #当输入真实值时返回Loss, 无真实标签，返回预测值
    def forward(self, x, y = None):
        # print(x.shape)  #torch.Size([1, 36]) --> 1*36  20 * 6
        x = self.embedding(x)  #(batch_size, sen_len)  -->  (batch_size, sen_len, vector_dim) 1*36*20
        '''
        embedding层的输入x
        tensor([[1, 2, 3],
                [2, 2, 0]])
        torch.Size([2, 3])
        输出：tensor([[[ 0.5128, -0.8417, -0.0978,  0.9089],
                      [-1.6924, -0.5557,  2.3441, -0.3055],
                      [-0.1817, -1.0516,  0.4169,  0.7366]],

                     [[-1.6924, -0.5557,  2.3441, -0.3055],
                      [-1.6924, -0.5557,  2.3441, -0.3055],
                      [ 0.9175,  0.0552,  0.6507, -0.5899]]], grad_fn=<EmbeddingBackward0>)
        torch.Size([2, 3, 4])
        '''
        # x = x.transpose(1,2)   #(batch_size, sen_len, vector_dim) --> (batch_size, vector_dim, sen_len)  1*20*36
        # print(self.pool(x).shape)  #torch.Size([1, 20, 6])  torch.Size([20, 20, 1])
        x = self.pool(x.transpose(1,2)).squeeze()  #(batch_size, vector_dim, sen_len) --> (batch_size, vector_dim, 1) --> (batch_size, vector_dim)
        # print(x.shape)  #torch.Size([20, 6])
        # print(x)
        '''
        tensor([-1.1193, -0.3366, -0.1084, -0.3166, -0.4468,  0.4557, -0.1072, -0.3137,
        -0.5630, -0.3415,  0.1695, -0.1391, -0.3830,  0.4779, -0.3354, -0.1998,
        -0.3323,  0.0043,  0.0594, -0.5142], grad_fn=<PermuteBackward0>)'''
        x = self.rnn_classifier(x)   #(batch_size, vector_dim) * (vector_dim, 1) -->  (batch_size, 1)
        # x = self.liner(x[0])
        y_pred = self.activate(x[0])  #(batch_size, 1) --> (batch_size, 1) x[0]: 6 * 1
        if y is not None:
            # print(y_pred.shape)             #torch.Size([20, 1])
            # print(y_pred.view(-1, 2).shape)  #torch.Size([10, 2])
            # print(torch.LongTensor(y.numpy()).view(-1).dtype)
            y = y.type(torch.int64)
            return self.loss(y_pred.view(-1, 2),y.view(-1))  #计算损失
        else:
            return y_pred


#构建词表 {"的"：0， "一"：1}
def build_vocab():
    chars = "的一是不了在人有我他这中大来上个国说们时你作会能好要就出最也"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char]= index  #将每一个汉字对用一个索引
    vocab['unk'] = len(vocab)   #处理未知字符
    return vocab
#构建单个数据样本
def build_samp(vocab, sentence_length):  #词表，获取长度
    # 随机从词表中获取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定那些字符出现为正样本
    if set("你我他") & set(x):
        y = 1
    else:
        y = 0
    #将取出的字转换成对应的序号，以便于后面向量化
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y
#构建数据集
def build_datasets(sample_length, vocab, sentence_length):
    datasets_x = []
    datasets_y = []
    for i in range(sample_length):
        x, y = build_samp(vocab, sentence_length)
        datasets_x.append(x)
        datasets_y.append([y])
    return torch.LongTensor(datasets_x), torch.FloatTensor(datasets_y)
#建立模型
def build_model(char_dim, sentence_length, vocab):
    model = TorchModel(char_dim, vocab, sentence_length)
    return model
#测试
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_datasets(200, vocab, sample_length)   #创建200个用于预测的样本
    print("本次共有%d个正样本，%d个负样本" % (sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        output = model(x)        #模型预测
        #判断0， 1哪个位置的概率更大
        _, y_pred = torch.max(output.data, 1)
        #print(torch.max(output.data, 1))
        for y_pred, y_true in zip(y_pred, y):
            if y_pred == 0 and int(y_true) == 0:
                correct += 1
            elif y_pred == 1 and int(y_true) == 1:
                correct += 1
            else:
                wrong += 1
    print("正确统计的个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)
#训练
def main():
    #参数配置
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    char_dim = 20  #每个字的维度
    sentence_length = 6   #样本的文本长度
    learning_rate = 0.005

    #构建词表
    vocab = build_vocab()
    #print(vocab)
    #建立模型
    model = build_model(char_dim, vocab, sentence_length)
    #选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    #开始训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_datasets(batch_size, vocab, sentence_length)  #构建训练样本
            optim.zero_grad()   #梯度清零
            loss = model(x, y)   #计算损失
            loss.backward()     #计算梯度
            optim.step()        #更新权重
            watch_loss.append(loss.item())
        print('-----\n第%d轮平均loss:%f' % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    #保存模型
    torch.save(model.state_dict(), './model/char_model.pth')
    #保存词表
    with open('./data/ch_vocab.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20   #每个字的维度
    sentence_length = 6  #样本文本长度
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    model = build_model(char_dim, vocab,sentence_length)
    model.load_state_dict(torch.load(model_path))  #加载模型

    #vocab.get()如果字符不在 vocab 字典中，则会返回默认值 vocab['unk']。
    x = [[vocab.get(char, vocab['unk']) for char in input_string] for input_string in input_strings]

    # x = []  #batch_size * 6
    # #序列化输入
    # for input_string in input_strings:
    #     x1 = []
    #     for char in input_string:
    #         flag = 0
    #         for key in vocab.keys():
    #             if char == key:
    #                 flag = 1
    #         if flag == 1:
    #             x1.append(vocab[char])
    #         else:
    #             x1.append(vocab['unk'])
    #     x.append(x1)
        # y.append([vocab[char] for char in input_string])
    # print(y)
    # print(torch.LongTensor(x).shape)
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(torch.LongTensor(x)) #预测
        # print(torch.max(y_pred.data, 1))
        # result = []
        # for i in y_pred.data.tolist():
        #     result.append(max(i))

        values, result = torch.max(y_pred.data, 1)  #返回概率和对应的索引
        # print(values)
    for i, input_string in enumerate(input_strings):

        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), values[i]) )

if __name__ == "__main__":
    main()
    test_strings = ['你我他这中大','今天天气很好', '出现警告信息', '深度学习基础', '腾讯阿里百度','他说查干湖好']
    predict("./model/char_model.pth", "./data/ch_vocab.json", test_strings)