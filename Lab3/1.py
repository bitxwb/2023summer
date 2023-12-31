import torch  # torch==1.7.1
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import re
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MAX_WORD = 10000  # 只保留最高频的10000词
MAX_LEN = 300  # 句子统一长度为300
word_count = {}  # 词-词出现的词数 词典


def drawtrainloss(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('train_loss_plot.png')
    plt.show()


def drawtestloss(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.savefig('test_loss_plot.png')
    plt.show()


# 清理文本，去标点符号，转小写
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()


# 分词方法
def tokenizer(sentence):
    return sentence.split()


#  数据预处理过程
def data_process(text_path, text_dir):  # 根据文本路径生成文本的标签

    print("data preprocess")
    file_pro = open(text_path, 'w', encoding='utf-8')
    for root, s_dirs, _ in os.walk(text_dir):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)  # 获取train和test文件夹下所有的路径
            text_list = os.listdir(i_dir)
            tag = os.path.split(i_dir)[-1]  # 获取标签
            if tag == 'pos':
                label = '1'
            if tag == 'neg':
                label = '0'
            if tag == 'unsup':
                continue

            for i in range(len(text_list)):
                if not text_list[i].endswith('txt'):  # 判断若不是txt,则跳过
                    continue
                f = open(os.path.join(i_dir, text_list[i]), 'r', encoding='utf-8')  # 打开文本
                raw_line = f.readline()
                pro_line = clean_str(raw_line)
                tokens = tokenizer(pro_line)  # 分词统计词数
                for token in tokens:
                    if token in word_count.keys():
                        word_count[token] = word_count[token] + 1
                    else:
                        word_count[token] = 0
                file_pro.write(label + ' ' + pro_line + '\n')
                f.close()
                file_pro.flush()
    file_pro.close()

    print("build vocabulary")

    vocab = {"<UNK>": 0, "<PAD>": 1}

    word_count_sort = sorted(word_count.items(), key=lambda item: item[1], reverse=True)  # 对词进行排序，过滤低频词，只取前MAX_WORD个高频词
    word_number = 1
    for word in word_count_sort:
        if word[0] not in vocab.keys():
            vocab[word[0]] = len(vocab)
            word_number += 1
        if word_number > MAX_WORD:
            break
    return vocab


# 定义Dataset
class MyDataset(Dataset):
    def __init__(self, text_path):
        file = open(text_path, 'r', encoding='utf-8')
        self.text_with_tag = file.readlines()  # 文本标签与内容
        file.close()

    def __getitem__(self, index):  # 重写getitem
        line = self.text_with_tag[index]  # 获取一个样本的标签和文本信息
        label = int(line[0])  # 标签信息
        text = line[2:-1]  # 文本信息
        return text, label

    def __len__(self):
        return len(self.text_with_tag)


# 根据vocab将句子转为定长MAX_LEN的tensor
def text_transform(sentence_list, vocab):
    sentence_index_list = []
    for sentence in sentence_list:
        sentence_idx = [vocab[token] if token in vocab.keys() else vocab['<UNK>'] for token in
                        tokenizer(sentence)]  # 句子分词转为id

        if len(sentence_idx) < MAX_LEN:
            for i in range(MAX_LEN - len(sentence_idx)):  # 对长度不够的句子进行PAD填充
                sentence_idx.append(vocab['<PAD>'])

        sentence_idx = sentence_idx[:MAX_LEN]  # 取前MAX_LEN长度
        sentence_index_list.append(sentence_idx)
    return torch.LongTensor(sentence_index_list)  # 将转为idx的词转为tensor


class LSTM(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, dropout):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size) # embedding层

        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=False,
                               dropout=dropout)
        self.decoder = nn.Linear(num_hiddens, 2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs的形状是（批量大小，词数），因此LSTM需要将序列长度（Seq_len）作为第一维，所以将输入转置后 再提取词特征
        embeddings = self.embedding(inputs.permute(1,0)) # permute(1,0)交换维度
        # LSTM只传入输入embeddings,因此只返回最后一层的隐藏层再各时间步的隐藏状态
        # outputs的形状是（词数，批量大小， 隐藏单元个数）
        outputs, _ = self.encoder(embeddings)
        outputs = self.dropout(outputs)
        # 连接初时间步和最终时间步的隐藏状态作为全连接层的输入。形状为(批量大小， 隐藏单元个数)
        encoding = outputs[-1] # 取LSTM最后一层结果
        outs = self.softmax(self.decoder(encoding)) # 输出层为二维概率[a,b]
        return outs


class GRU(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, dropout):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)  # embedding层

        self.encoder = nn.GRU(input_size=embed_size,
                              hidden_size=num_hiddens,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=False)
        self.decoder = nn.Linear(num_hiddens, 2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings)
        outputs = self.dropout(outputs)
        encoding = outputs[-1]
        outs = self.softmax(self.decoder(encoding))
        return outs


# 模型训练
# 模型训练
def train(model, train_data, test_data, vocab, epoches):
    # print(len(train_data))
    # print(len(test_data))
    print('train model')
    model = model.to(device)
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    train_losses = []  # 用于保存训练集上的损失
    train_accs = []  # 用于保存训练集上的准确率
    test_losses = []  # 用于保存测试集上的损失
    test_accs = []  # 用于保存测试集上的准确率
    best_acc = 0.0
    for epoch in range(epoches):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for idx, (text, label) in enumerate(train_data):
            train_x = text_transform(text, vocab).to(device)
            train_y = label.to(device)

            optimizer.zero_grad()
            pred = model(train_x)
            loss = criterion(pred.log(), train_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += accuracy(pred, train_y)

        avg_loss = running_loss / 25000  # 平均训练损失
        avg_acc = running_acc / len(train_data)  # 平均训练准确率
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        print("train_avg_loss:", avg_loss, " train_avg_acc:", avg_acc)

        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for idx, (text, label) in enumerate(test_data):
                test_x = text_transform(text, vocab).to(device)
                test_y = label.to(device)

                pred = model(test_x)
                test_loss += criterion(pred.log(), test_y).item()
                test_acc += accuracy(pred, test_y)

        avg_test_loss = test_loss / 25000  # 平均测试损失
        avg_test_acc = test_acc / len(test_data)  # 平均测试准确率
        test_losses.append(avg_test_loss)
        test_accs.append(avg_test_acc)
        print("test_avg_loss:", avg_test_loss, " test_avg_acc:", avg_test_acc)

        model.train()
        # 保存训练完成后的模型参数
        if avg_test_acc > best_acc:
            best_acc = avg_test_acc
            torch.save(model.state_dict(), 'IMDB_parameter.pkl')

    drawtrainloss(train_losses)
    drawtestloss(test_losses)


def test(model, test_data, vocab):
    print('test model')
    model = model.to(device)
    # model.eval()
    positive_correct = 0
    positive_total = 0
    negative_correct = 0
    negative_total = 0

    for idx, (text, label) in enumerate(test_data):
        train_x = text_transform(text, vocab).to(device)
        train_y = label.to(device)
        pred = model(train_x)
        predicted_labels = pred.argmax(axis=1)
        true_labels = label.squeeze()

        for true_label, predicted_label in zip(true_labels, predicted_labels):
            if true_label == 1:
                positive_correct += (predicted_label == true_label).item()
                positive_total += 1
            else:
                negative_correct += (predicted_label == true_label).item()
                negative_total += 1
    positive_acc = positive_correct / positive_total if positive_total > 0 else 0
    negative_acc = negative_correct / negative_total if negative_total > 0 else 0
    total_acc = (positive_correct + negative_correct) / (positive_total + negative_total) if (
                                                                                                     positive_total + negative_total) > 0 else 0
    return positive_acc, negative_acc, total_acc


# 计算预测准确性
def accuracy(y_pred, y_true):
    label_pred = y_pred.max(dim=1)[1]
    acc = len(y_pred) - torch.sum(torch.abs(label_pred - y_true))  # 正确的个数
    return acc.detach().cpu().numpy() / len(y_pred)


def main():
    train_dir = './aclImdb/train'  # 原训练集文件地址
    train_path = './train.txt'  # 预处理后的训练集文件地址

    test_dir = './aclImdb/test'  # 原训练集文件地址
    test_path = './test.txt'  # 预处理后的训练集文件地址

    vocab = data_process(train_path, train_dir)  # 数据预处理
    data_process(test_path, test_dir)
    np.save('vocab.npy', vocab)  # 词典保存为本地
    vocab = np.load('vocab.npy', allow_pickle=True).item()  # 加载本地已经存储的vocab

    # 构建MyDataset实例
    train_data = MyDataset(text_path=train_path)
    test_data = MyDataset(text_path=test_path)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    # 生成模型
    # model = LSTM(vocab=vocab, embed_size=300, num_hiddens=128, num_layers=2, dropout=0.1)  # 定义模型
    model = GRU(vocab=vocab, embed_size=300, num_hiddens=128, num_layers=2, dropout=0.1)  # 定义模型
    train(model=model, train_data=train_loader, test_data=test_loader, vocab=vocab, epoches=30)

    # 加载训练好的模型
    model.load_state_dict(torch.load('IMDB_parameter.pkl',
                                     map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))

    # 测试结果
    positive_acc, negative_acc, total_acc = test(model=model, test_data=test_loader, vocab=vocab)
    print(f"positive_acc = {positive_acc}.")
    print(f"negative_acc = {negative_acc}.")
    print(f"total_acc = {total_acc}.")


if __name__ == '__main__':
    main()
