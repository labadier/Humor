import pandas as pd, numpy as np, os, csv, time, random
from sklearn.metrics import f1_score, accuracy_score, classification_report
from utils.params import params, bcolors
from matplotlib import pyplot as plt

from googletrans import Translator 

def load_data(filename, lang):

  dataframe = pd.read_csv(filename, dtype=str)
  if lang == 'es':
    dataframe = dataframe[dataframe['source'] == 'Haha']
  elif lang == 'en':
    dataframe = dataframe[dataframe['source'] != 'Haha']

  data = {key:dataframe[key].to_numpy() if key != 'humor' else dataframe['humor'].astype(int).to_numpy() for key in dataframe.columns}
  return data

def plot_training(history, model, measure='loss'):
 
    plt.plot(history[measure])
    plt.plot(history['dev_' + measure])
    plt.legend(['train', 'dev'], loc='upper left')
    plt.ylabel(measure)
    plt.xlabel('Epoch')
    if measure == 'loss':
        x = np.argmin(history['dev_loss'])
    else: x = np.argmax(history['dev_acc'])

    plt.plot(x,history['dev_' + measure][x], marker="o", color="red")

    if os.path.exists('logs') == False:
        os.system('mkdir logs')

    plt.savefig( f'logs/train_history_{model}.png')

def mergeData(mode) -> None:
  with open(os.path.join(f'data/{mode}.csv'), 'wt', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['source', 'index', 'text', 'humor'])

    #Haha
    data = pd.read_csv(f'data/Haha/{mode}.csv')
    for row in data.iterrows():
      spamwriter.writerow(['Haha', row[1]['id'], row[1]['text'].replace('\n', ' '), row[1]['is_humor']])

    #Hahackathon
    data = pd.read_csv(f'data/HaHackathon/{mode}.csv')
    for row in data.iterrows():
      spamwriter.writerow(['HaHackathon', row[1]['id'], row[1]['text'].replace('\n', ' '), row[1]['is_humor']])

    #joker
    data = pd.read_csv(f'data/JOKER/Task 1/{mode}/joker_task1_en_{mode}.csv')
    for row in data.iterrows():
      spamwriter.writerow(['joker', row[1]['ID'], row[1]['WORDPLAY'].replace('\n', ' '), 1])

    #headlines
    data = pd.read_csv(f'data/semeval-2020-task-7-dataset-headlines/{mode}.csv')
    for row in data.iterrows():
      if float(row[1]['meanGrade']) > 2.0:
        trans =  row[1]['original'].replace('\n', ' ')        
        spamwriter.writerow(['headlines', row[1]['id'], trans.replace('<', '').replace('/>', ''), 0])
        b = trans.find('<')
        e = trans.find('>')
        spamwriter.writerow(['headlines', row[1]['id']*2, trans[:b] + row[1]['edit'].replace('\n', ' ') + trans[e+1:], 1])

def TranslatePivotLang(sourceFile = 'data/train.csv', outputFile = 'train', step=29) -> None:

  print(f'Pivot Language: 0%\r', end="")

  perc = 0
  data_frame = pd.read_csv(sourceFile, dtype=str)

  with open(f'data/{outputFile}_inverted.csv', 'wt', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(list(data_frame.columns))
    
    for i in range(0, len(data_frame), step):
      
      if (i*100.0)/len(data_frame) - perc > 0.25:
        perc = (i*100.0)/len(data_frame)
        print(f'Pivot Language: {perc:.2f}%\t\r', end = "")

      data = data_frame[i:i + step].copy() 

      if len(set(data['source'].to_list())) == 1:
        ts = Translator()
        time.sleep(random.random()*3)
        try:
          data['text'] = (ts.translate(text='\n'.join(data['text'].to_list()), 
                              src = 'es' if (data.iloc[0]['source'] == 'Haha') else 'en', 
                              dest = 'en' if (data.iloc[0]['source'] == 'Haha') else 'es').text).split('\n')
        except:
          print(f'An exception occurred on index {i}')

      else:
        ts = Translator()
        for j in range(step):
          data.iloc[j]['text'] = ts.translate(text=data.iloc[j]['text'], 
                        src = 'es' if (data.iloc[j]['source'] == 'Haha') else 'en',
                        dest = 'en' if (data.iloc[j]['source'] == 'Haha') else 'es').text
          time.sleep(random.random()*3)

      for j in data.iterrows():
        spamwriter.writerow(j[1].to_list())
        
    print(f'\rPivot Language : 100%\t')

def backTranslation(sourceFile = 'data/train_inverted.csv', step=29) -> None:

  print(f'Pivot Language: 0%\r', end="")

  perc = 0
  data_frame = pd.read_csv(sourceFile, dtype=str)

  with open(f"{sourceFile.split('_')[0]}_back.csv", 'wt', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(list(data_frame.columns))
    
    for i in range(0, len(data_frame), step):
      
      if (i*100.0)/len(data_frame) - perc > 0.25:
        perc = (i*100.0)/len(data_frame)
        print(f'Back Language: {perc:.2f}%\t\r', end = "")

      data = data_frame[i:i + step].copy() 

      if len(set(data['source'].to_list())) == 1:
        ts = Translator()
        time.sleep(random.random()*3)
        try:
          data['text'] = (ts.translate(text='\n'.join(data['text'].to_list()), 
                              src = 'es' if not (data.iloc[0]['source'] == 'Haha') else 'en', 
                              dest = 'en' if not (data.iloc[0]['source'] == 'Haha') else 'es').text).split('\n')
        except:
          print(f'An exception occurred on index {i}')

      else:
        ts = Translator()
        for j in range(step):
          data.iloc[j]['text'] = ts.translate(text=data.iloc[j]['text'], 
                        src = 'es' if not (data.iloc[j]['source'] == 'Haha') else 'en',
                        dest = 'en' if not (data.iloc[j]['source'] == 'Haha') else 'es').text
          time.sleep(random.random()*3)

      for j in data.iterrows():
        spamwriter.writerow(j[1].to_list())
        
    print(f'\rBack Language : 100%\t')


def evaluate(file_path):


  sources = ['HaHackathon', 'joker', 'hedlines', 'Haha']

  file = pd.read_csv(file_path)

  for i in sources:

     
    data = file[file['source'] == i]
    if not len(data):
      continue
  
    print(f"{bcolors.OKGREEN}{bcolors.BOLD}== {i} Report == {bcolors.ENDC}")

    y_hat = data['prediction' if 'prediction' in data.columns else 'is_humor'].astype(int).to_numpy()
    y = data['ground_humor'].astype(int).to_numpy()
    print(classification_report(y, y_hat, target_names=['non-humor', 'humor'],  digits=3, zero_division=1))

  print(f"{bcolors.OKBLUE}{bcolors.BOLD}{'='*10}\n== Overall Report == {bcolors.ENDC}") 

  y_hat = file['prediction' if 'prediction' in file.columns else 'is_humor'].astype(int).to_numpy()
  y = file['ground_humor'].astype(int).to_numpy()
  print(classification_report(y, y_hat, target_names=['non-humor', 'humor'],  digits=3, zero_division=1))
