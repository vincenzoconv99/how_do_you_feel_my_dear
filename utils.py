import pandas as pd
import os
import re
import neattext as nt
from neattext.explainer import emoji_explainer
from neattext import TextExtractor
import numpy as np
import tensorflow as tf

def load_emodec_tweets(path='./datasets/Emodec_tweets/tweet_emotions.csv'):
    emodec = pd.read_csv(path)
    return emodec.drop('tweet_id', axis=1)

def load_wassa(path='./datasets/Wassa/'):
    wassa_data = []
    for file in os.listdir(path):
        with open(path+'/'+file, mode='r', encoding='utf8') as f:
            file_lines = f.readlines()
        #dataset_type = file.split('.')[1]
        for line in file_lines:
            splitted_line = line.split('\t')
            try:
                emotion_score = float(splitted_line[-1])
                text = splitted_line[-3]
                emotion = splitted_line[-2]
                
                #if emotion_score > 0.5: #if emotion_score >= 0.9 and emotion in {'anger', 'sadness'}:
                wassa_data.append([emotion, text])
            except:
                pass
    return pd.DataFrame(wassa_data, columns = ['sentiment', 'content'])


def normalize_text(text):
    extr = TextExtractor(text=text)
    emojis=[]
    if len(extr.extract_emojis())>0:
        emojis = [ *extr.extract_emojis()[0] ] # extracting emojis
        emojis_expl = [emoji_explainer(e) for e in emojis] # explaining emojis
    
    text = re.sub(r'\d+','',text) # removing numbers
    docx = nt.TextFrame(text)
    docx.remove_stopwords(lang='en') # removing stop words
    docx.remove_emails() #removing emails
    docx.remove_userhandles() #removing usernames
    #docx.remove_special_characters() # removing special characters
    text = docx.text
    if len(emojis) > 0:
        try:
            docx.remove_emojis() # removing the emojis and subtitute them with explanations
            text = docx.text
            text += ' '.join(emojis_expl)
        except:
            pass
        
    text = text.lower() # lowering case
    text = text.replace('_',' ')
    text = re.sub(' +', ' ', text) # replacing multiple spaces with single
    return text

def ohe(index):
    ohe_vector = [0]*5
    ohe_vector[index] = 1
    return ohe_vector

def create_tf_dataset(data, BATCH_SIZE):
    sentiments = ['fear', 'joy', 'anger', 'surprise', 'sadness']
    features = []
    labels = []

    for x in data.iterrows():
        features.append(tf.constant(x[1]['content']))
        labels.append(tf.constant( ohe(sentiments.index(x[1]['sentiment'])) ))

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset


def classification_report(metrics):
    print('accuracy mean', np.mean(metrics['accuracy']))
    print('accuracy std', np.std(metrics['accuracy']))
    print('f1_score_weighted mean', np.mean(metrics['f1_score_weighted']))
    print('f1_score_weighted std', np.std(metrics['f1_score_weighted']))
    print('f1_score_micro mean', np.mean(metrics['f1_score_micro']))
    print('f1_score_micro std', np.std(metrics['f1_score_micro']))
    print('f1_score_macro mean', np.mean(metrics['f1_score_macro']))
    print('f1_score_macro std', np.std(metrics['f1_score_macro']))