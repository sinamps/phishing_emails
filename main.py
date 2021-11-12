import sys
import json
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import time
# from netvlad.mynextvlad import NeXtVLAD
# from netvlad.internetnextvlad import NextVLAD

TRAIN_PATH = "/s/bach/h/proj/COVID-19/smps/phishing/phishing_dataset_trainset.csv"
# TRAIN_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/small_shuf.jsonl"
# VAL_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/val1000_shuf.jsonl"
# VAL_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/small_shuf.jsonl"
TEST_PATH = "/s/bach/h/proj/COVID-19/smps/phishing/phishing_dataset_testset.csv"
# TEST_PATH = "/s/bach/h/proj/COVID-19/claim/pycharm_projects/version1/datasets/irony/figlang/small_shuf.jsonl"
SAVE_PATH = "/s/lovelace/c/nobackup/iray/sinamps/phishing/"
# LOAD_PATH = "/s/lovelace/c/nobackup/iray/sinamps/claim_project/2021-06-01_models/jun10_tbinv_ensemble_with_bigger_batch-final-epoch-15"

PRE_TRAINED_MODEL = 'bert-large-cased'
MAXTOKENS = 512
NUM_EPOCHS = 8
BERT_EMB = 1024
H = 512
BS = 4
INITIAL_LR = 1e-6
save_epochs = []

CUDA_0 = 'cuda:0'
CUDA_1 = 'cuda:1'
CUDA_2 = 'cuda:1'


def ensemble_data(dataset, context_size=3):
    new_dataset = []
    for line in dataset:
        data = json.loads(line)
        if len(data['context']) > context_size:
            data['context'] = data['context'][-context_size:]

        for i in range(min(context_size, len(data['context']))):
            d = {'label': data['label'],
                 'response': data['response'],
                 'context': data['context'][i:]}

            new_dataset.append(d)

    return new_dataset



def myprint(mystr, logfile):
    print(mystr)
    print(mystr, file=logfile)


def load_data(file_name):
    # Load and prepare the data
    texts = []
    labels = []
    try:
        df = pd.read_csv(file_name)
        # f = open(file_name)
    except:
        print('my log: could not read file')
        exit()
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def evaluate_model(labels, predictions, titlestr, logfile):
    myprint(titlestr, logfile)
    conf_matrix = confusion_matrix(labels, predictions)
    myprint("Confusion matrix- \n" + str(conf_matrix), logfile)
    # print("Confusion matrix- \n", conf_matrix)
    # print("Confusion matrix- \n", conf_matrix, file=logfile)
    acc_score = accuracy_score(labels, predictions)
    myprint('  Accuracy Score: {0:.2f}'.format(acc_score), logfile)
    # print('  Accuracy Score: {0:.2f}'.format(acc_score))
    # print('  Accuracy Score: {0:.2f}'.format(acc_score), file=logfile)
    myprint('Report', logfile)
    cls_rep = classification_report(labels, predictions)
    myprint(cls_rep, logfile)
    # print(cls_rep)
    # print(cls_rep, file=logfile)


def feed_model(model, data_loader):
    outputs_flat = []
    labels_flat = []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(CUDA_0)
        attention_mask = batch['attention_mask'].to(CUDA_0)
        outputs = model(input_ids, attention_mask=attention_mask)
        outputs = outputs.detach().cpu().numpy()
        labels = batch['labels'].to('cpu').numpy()
        outputs_flat.extend(np.argmax(outputs, axis=1).flatten())
        labels_flat.extend(labels.flatten())
        del outputs, labels, attention_mask, input_ids
    return labels_flat, outputs_flat


class MyInternetModel(nn.Module):
    def __init__(self, base_model, n_classes, dropout=0.05):
        super().__init__()

        self.base_model = base_model.to(CUDA_0)
        self.final = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(BERT_EMB, BERT_EMB),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(BERT_EMB, n_classes)
        ).to(CUDA_2)

    def forward(self, input_, **kwargs):
        X = input_
        if 'attention_mask' in kwargs:
            attention_mask = kwargs['attention_mask']
        else:
            print("my err: attention mask is not set, error maybe")
        hidden_states = self.base_model(X.to(CUDA_0), attention_mask=attention_mask.to(CUDA_0)).last_hidden_state
        cls = hidden_states[:, 0, :]
        myo = self.final(cls.to(CUDA_2))
        myo = nn.functional.softmax(myo, dim=1)
        return myo


if __name__ == '__main__':
    args = sys.argv
    epochs = NUM_EPOCHS
    logfile = open('log_file_' + args[0].split('/')[-1][:-3] + str(time.time()) + '.txt', 'w')
    myprint(INITIAL_LR, logfile)
    train_texts, train_labels = load_data(TRAIN_PATH)
    # val_texts, val_labels = load_data(VAL_PATH)
    test_texts, test_labels = load_data(TEST_PATH)
    # tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL, {'model_max_length': MAXTOKENS})
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
    # print(tokenizer.model_max_length)
    tokenizer.model_max_length = MAXTOKENS
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    # val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    train_dataset = MyDataset(train_encodings, train_labels)
    # val_dataset = MyDataset(val_encodings, val_labels)
    test_dataset = MyDataset(test_encodings, test_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL)
    model = MyInternetModel(base_model=base_model, n_classes=2)
    # model.load_state_dict(torch.load(LOAD_PATH))
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

    optim = AdamW(model.parameters(), lr=INITIAL_LR)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=2000,
                                                num_training_steps=total_steps)
    loss_model = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print(' EPOCH {:} / {:}'.format(epoch+1, epochs))
        outputs_flat = []
        labels_flat = []
        for step, batch in enumerate(train_loader):
            if step % 100 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))
            optim.zero_grad()
            input_ids = batch['input_ids'].to(CUDA_0)
            attention_mask = batch['attention_mask'].to(CUDA_0)
            labels = batch['labels'].to(CUDA_2)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_model(outputs, labels)
            loss.backward()
            optim.step()
            scheduler.step()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            outputs_flat.extend(np.argmax(outputs, axis=1).flatten())
            labels_flat.extend(labels.flatten())
            del outputs, labels, attention_mask, input_ids
        evaluate_model(labels_flat, outputs_flat, 'Train set Result epoch ' + str(epoch+1), logfile)
        del labels_flat, outputs_flat
        model.eval()
        # val_labels, val_predictions = feed_model(model, val_loader)
        # evaluate_model(val_labels, val_predictions, 'Validation set Result epoch ' + str(epoch+1), logfile)
        # del val_labels, val_predictions
        if (epoch+1) in save_epochs:
            test_labels, test_predictions = feed_model(model, test_loader)
            evaluate_model(test_labels, test_predictions, 'Test set Result epoch ' + str(epoch+1), logfile)
            del test_labels, test_predictions
            try:
                torch.save(model.state_dict(), (SAVE_PATH + args[0].split('/')[-1][:-3] + '-auto-' + str(epoch+1)))
            except:
                myprint("Could not save the model", logfile)
        model.train()
    del train_loader
    model.eval()
    myprint('--------------Training complete--------------', logfile)
    torch.save(model.state_dict(), SAVE_PATH + args[0].split('/')[-1][:-3] + '-final')
    test_labels, test_predictions = feed_model(model, test_loader)
    evaluate_model(test_labels, test_predictions, 'Final Testing', logfile)
    del test_labels, test_predictions
