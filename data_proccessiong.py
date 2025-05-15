import pandas as pd
import math
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from itertools import chain
import time
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score


class SAData(Dataset):
    """
    对数据集进行导入，根据train参数对数据集进行分割
    """

    def __init__(self, train):
        self.train = train
        self.data = pd.read_csv("/kaggle/input/dataset/train.tsv", sep='\t')
        if self.train:
            self.data = self.data.sample(frac=0.8, replace=False, random_state=42, axis=0)
            self.data = self.data.reset_index(drop=True)
            self.len = self.data.shape[0]
        else:
            self.data = self.data.sample(frac=0.2, replace=False, random_state=42, axis=0)
            self.data = self.data.reset_index(drop=True)
            self.len = self.data.shape[0]
        self.x_data, self.y_data = self.data['Phrase'], self.data['Sentiment']

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


train_set = SAData(train=True)
test_set = SAData(train=False)

BATCH_SIZE = 32
USE_GPU = True
N_EPOCH = 10

train_loader = DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validiation_loader = DataLoader(
    dataset=test_set,
    batch_size=BATCH_SIZE,
    shuffle=False
)


def create_tensor(tensor):
    if USE_GPU:
        device = torch.device('cuda')
        tensor = tensor.to(device)
        return tensor
    else:
        return tensor


def phrase2list(phrase):
    arr = [ord(c) for c in phrase]
    return arr, len(arr)


def make_tensor(phrase, sentiment):
    sequences_and_len = [phrase2list(phrase) for phrase in phrase]
    sentiment = sentiment.long()
    phrase_sequences = [sl[0] for sl in sequences_and_len]
    phrase_len = torch.LongTensor([sl[1] for sl in sequences_and_len])

    # 组织张量，不满足长度的用0填充
    time_tensor = torch.zeros(len(phrase_sequences), phrase_len.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(phrase_sequences, phrase_len)):
        time_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # 为了加速RNN处理，在此处降序排列
    seq_len, index = phrase_len.sort(dim=0, descending=True)
    time_tensor = time_tensor[index]
    sentiment = sentiment[index]

    return create_tensor(time_tensor), create_tensor(seq_len), create_tensor(sentiment)


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, bidirectional=True, dropout=0.5):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layers = n_layers
        self.n_direction = 2 if bidirectional else 1

        # 网络层
        self.embedding = torch.nn.Embedding(self.input_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                                bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_direction, output_size)

    # 初始化隐藏层
    def __init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_direction, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        input = input.t()
        batch_size = input.size(1)
        hidden = self.__init_hidden(batch_size)
        embedding = self.embedding(input)
        embedding = self.dropout(embedding)
        gru_input = pack_padded_sequence(embedding, seq_lengths.cpu())
        output, hidden = self.gru(gru_input, hidden)
        if self.n_direction == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = torch.cat(hidden[-1])
        fc_output = self.fc(hidden_cat)
        return fc_output


def train_epoch(model, dataloader, criterion, optimizer, scheduler=None):
    model.train()
    train_loss = 0
    for i, (phrase, sentiment) in enumerate(dataloader):
        inputs, lengths, target = make_tensor(phrase, sentiment)
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        train_loss += loss.item()
        if i % 100 == 0:
            print(f"batch:{i:02d} loss:{loss.item():.4f} total_loss:{train_loss:.4f}")
        if scheduler:
            scheduler.step()


def evaluate(model, dataloader, criterion):
    model.eval()
    eva_loss = 0
    all_preds = []
    all_target = []
    with torch.no_grad():
        for phrase, sentiment in dataloader:
            inputs, lengths, target = make_tensor(phrase, sentiment)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, target)
            eva_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_target.extend(target.cpu().numpy())
    acc = accuracy_score(all_target, all_preds)
    return eva_loss / len(dataloader), acc


def get_test():
    test_st = pd.read_csv('/kaggle/input/dataset/test.tsv', sep='\t')
    test_st['Phrase'] = test_st['Phrase'].astype(str)
    PhraseID = test_st['PhraseId']
    Phrase = test_st['Phrase']
    return PhraseID, Phrase


def make_tensor_test(phrase):
    sequences_and_len = [phrase2list(phrase) for phrase in phrase]
    phrase_sequences = [sl[0] for sl in sequences_and_len]
    phrase_len = torch.LongTensor([sl[1] for sl in sequences_and_len])

    # 组织张量，不满足长度的用0填充
    time_tensor = torch.zeros(len(phrase_sequences), phrase_len.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(phrase_sequences, phrase_len)):
        time_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # 为了加速RNN处理，在此处降序排列
    seq_len, index = phrase_len.sort(dim=0, descending=True)
    time_tensor = time_tensor[index]
    _, org_index = index.sort(descending=False)

    return create_tensor(time_tensor), create_tensor(seq_len), org_index


def predict():
    PhraseID, Phrase = get_test()
    predict_list = []
    model = torch.load('/kaggle/working/models/my_model.pkl')
    batchnum = math.ceil(PhraseID.shape[0] / BATCH_SIZE)

    with torch.no_grad():
        for i in range(batchnum):
            if i == batchnum - 1:
                phraseBatch = Phrase[BATCH_SIZE * i:]
            else:
                phraseBatch = Phrase[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
            inputs, lengths, org_index = make_tensor_test(phraseBatch)
            output = model(inputs, lengths)
            sentiment = output.max(dim=1, keepdim=True)[1]  # 修正维度索引
            sentiment = sentiment[org_index].squeeze(dim=1)  # 修正squeeze调用
            predict_list.append(sentiment.cpu().numpy().tolist())

    predict_list = list(chain.from_iterable(predict_list))
    result = pd.DataFrame({'PhraseId': PhraseID, 'Sentiment': predict_list})
    result.to_csv('/kaggle/working/submission.csv', index=False)
    print("submission生成成功")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNClassifier(
        input_size=128,
        hidden_size=128,
        output_size=len(set(train_set.y_data)),
        n_layers=2
    ).to(device)

    # 创建子文件夹
    models_dir = '/kaggle/working/models'
    os.makedirs(models_dir, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_acc = 0
    for epoch in range(0, N_EPOCH + 1):
        print(f"Epoch:{epoch:02d}/{N_EPOCH:02d}")
        print("-" * 60)
        train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, validiation_loader, criterion)
        print(f"now accuracy:{val_acc:.4f}  best accuracy:{best_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存模型
            model_path = os.path.join(models_dir, 'my_model.pkl')
            torch.save(model, model_path)
    predict()

if __name__ == '__main__':
    main()