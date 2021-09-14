import os
import email
import re
from email import policy, parser
from email_reply_parser import EmailReplyParser
import talon
from talon import signature
import pandas as p
import ftfy
from bs4 import BeautifulSoup

DIR = 'datasets/IWSPA 2.0 Train/'
# FH_DIR = 'datasets/IWSPA 2.0 Train/IWSPA2.0_Training_Full_Header'
# NH_DIR = 'datasets/IWSPA 2.0 Train/IWSPA2.0_Training_No_Header'


def clean_remove_header(f):
    email_msg = email.message_from_file(f, policy=policy.default)
    # email_msg = parser.Parser(policy=policy.default).parse(fp=f)
    # print(email_msg.get_body())
    # print(email_msg['From'])
    if email_msg.is_multipart():
        # I need to fix this later so that we do not lose these three phishing samples
        # print(f.name)
        return None
    else:
        payload = email_msg.get_payload()
    # payload = email_msg.get_payload()
    text = EmailReplyParser.parse_reply(payload)
    text = talon.quotations.extract_from_plain(text)
    text, sign = signature.bruteforce.extract_signature(text)
    text, sign = signature.extract(text, sender='')
    # text = ftfy.fix_text(text)
    # print(text)
    # print('-----------------------------------------------------------------')
    # print('-----------------------------------------------------------------')
    return text


def apply_conditions(text):
    txt = ftfy.fix_text(text)
    txt = BeautifulSoup(txt, "lxml").text
    txt = re.sub(".[^\s]{26,}", '', txt)
    txt = re.sub('\n+', '\n', txt)
    if re.search("class=3Dad_group", txt):
        return None
    txt = re.sub("download>=20", '', txt)
    txt = re.sub("=09\n", '', txt)
    txt = re.sub("<<>", '', txt)
    txt = re.sub("<>", '', txt)
    txt = re.sub('=\d', '', txt)
    if len(txt) < 25:
        return None
    return txt


def get_data(dirname, remove_header, fc, isPhish):
    texts_list = []
    for filename in os.listdir(dirname):
        f = open(os.path.join(dirname, filename), 'r')  # open in readonly mode
        fc = fc + 1
        if remove_header:
            text = clean_remove_header(f)
            if text is None:
                f.close()
                continue
        else:
            text = f.read()
        txt = apply_conditions(text)
        if not txt:
            # print('text is none')
            f.close()
            continue
        texts_list.append(txt)
        f.close()
    labels_list = [(1 if isPhish else 0) for i in range(len(texts_list))]
    return texts_list, labels_list, fc


def read_files(input_dir):
    texts = []
    labels = []
    fc = 0
    for dirname in os.listdir(input_dir):
        remove_header = False
        if dirname.endswith('Full_Header'):
            remove_header = True
        legit_dir = input_dir + dirname + '/legit/'
        phish_dir = input_dir + dirname + '/phish/'
        legit_texts, legit_labels, fc = get_data(legit_dir, remove_header, fc, isPhish=False)
        phish_texts, phish_labels, fc = get_data(phish_dir,  remove_header, fc, isPhish=True)
        texts.extend(legit_texts)
        texts.extend(phish_texts)
        labels.extend(legit_labels)
        labels.extend(phish_labels)
    dict = {'text': texts, 'label': labels}
    df = p.DataFrame(data=dict)
    # print(fc)
    # print(df)
    df.to_csv('phishing_dataset.csv', encoding='utf-8')


if __name__ == '__main__':
    talon.init()
    read_files(DIR)
