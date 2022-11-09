import torch, os, random, csv
from utils.params import params
import numpy as np, pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from utils.params import params, bcolors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

class Data(Dataset):

  def __init__(self, data):

    self.data = data
    
  def __len__(self):
    return len(self.data['text'])

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    ret = {key: self.data[key][idx] for key in self.data.keys()}
    return ret
   

def HugginFaceLoad(language, weigths_source):

  prefix = 'data' if weigths_source == 'offline' else ''
  model = AutoModel.from_pretrained(os.path.join(prefix , params.models[language]))
  tokenizer = AutoTokenizer.from_pretrained(os.path.join(prefix , params.models[language]), do_lower_case=True, TOKENIZERS_PARALLELISM=True)

  return model, tokenizer

def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

class SeqModel(torch.nn.Module):

  def __init__(self, interm_size, max_length, **kwargs):

    super(SeqModel, self).__init__()
		
    self.mode = kwargs['mode']
    self.best_acc = None
    self.lang = kwargs['lang']
    self.max_length = max_length
    self.interm_neurons = interm_size
    self.transformer, self.tokenizer = HugginFaceLoad( kwargs['lang'], self.mode)
    self.intermediate = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=self.interm_neurons), torch.nn.LeakyReLU(),
                                            torch.nn.Linear(in_features=self.interm_neurons, out_features=self.interm_neurons>>1),
                                            torch.nn.LeakyReLU())
    
    self.classifier = torch.nn.Linear(in_features=self.interm_neurons>>1, out_features=2)
    self.loss_criterion = torch.nn.CrossEntropyLoss()
    
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, data, get_encoding=False):

    ids = self.tokenizer(data, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)

    X = self.transformer(**ids)[0]

    X = X[:,0]
    enc = self.intermediate(X)
    output = self.classifier(enc)
    if get_encoding == True:
      return enc

    return output 

  def load(self, path):
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Weights Loaded{bcolors.ENDC}") 
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)

  def evaluate(self, data, batch_size, wp, outputFile, encode):
    
    devloader = DataLoader(Data(data), batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)
    self.eval()
    self.load(os.path.join(wp))
    
    with open(outputFile, 'wt', newline='', encoding="utf-8") as csvfile:
      spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      spamwriter.writerow(['source', 'index', 'encoding' if encode == True else 'prediction', 'ground_humor'])

      with torch.no_grad():
        for batch in devloader:   
          dev_out = self.forward(batch['text'], get_encoding=encode).cpu().numpy()
          for i in range(len(batch['text'])):
            spamwriter.writerow([batch[key][i] if key != 'text' else (' '.join([str(j) for j in dev_out[i]]) if encode else np.argmax(dev_out[i])) for key in batch.keys()])
      


  def makeOptimizer(self, lr=1e-5, decay=2e-5, multiplier=1, increase=0.1):

    # if self.lang == 'fr':
    #   return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)

    params = []
    for l in self.transformer.encoder.layer:

      params.append({'params':l.parameters(), 'lr':lr*multiplier}) 
      multiplier += increase

    try:
      params.append({'params':self.transformer.pooler.parameters(), 'lr':lr*multiplier})
    except:
      print(f'{bcolors.WARNING}Warning: No Pooler layer found{bcolors.ENDC}')

    params.append({'params':self.intermediate.parameters(), 'lr':lr*multiplier})
    params.append({'params':self.classifier.parameters(), 'lr':lr*multiplier})

    return torch.optim.RMSprop(params, lr=lr*multiplier, weight_decay=decay)

def sigmoid( z ):
  return 1./(1 + torch.exp(-z))

def compute_acc(ground_truth, predictions):
  out = torch.max(predictions, 1).indices.detach().cpu().numpy()

  return f1_score(out, ground_truth.detach().cpu().numpy())


def train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, split=1):
  
  eloss, eacc, edev_loss, edev_acc = [], [], [], []

  optimizer = model.makeOptimizer(lr=lr, decay=decay)
  batches = len(trainloader)

  for epoch in range(epoches):

    running_loss = 0.0
    perc = 0
    acc = 0
    
    model.train()
    last_printed = ''

    for j, data in enumerate(trainloader, 0):

      torch.cuda.empty_cache()         
      labels = data['humor'].to(model.device)     
      
      optimizer.zero_grad()
      outputs = model(data['text'])
      loss = model.loss_criterion(outputs, labels)
   
      loss.backward()
      optimizer.step()

      # print statistics
      with torch.no_grad():
        if j == 0:
          acc = compute_acc(labels, outputs)
          running_loss = loss.item()
        else: 
          acc = (acc + compute_acc(labels, outputs))/2.0
          running_loss = (running_loss + loss.item())/2.0

      if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
        
        perc = (1+j)*100.0/batches
        last_printed = f'\rEpoch:{epoch+1:3d} of {epoches} step {j+1} of {batches}. {perc:.1f}% loss: {running_loss:.3f}'
        
        print(last_printed , end="")#+ compute_eta(((time.time()-start_time)*batches)//(j+1))

    model.eval()
    eloss.append(running_loss)
    with torch.no_grad():
      out = None
      log = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 
        labels = data['humor'].to(model.device) 

        dev_out = model(data['text'])
        if k == 0:
          out = dev_out
          log = labels
        else: 
          out = torch.cat((out, dev_out), 0)
          log = torch.cat((log, labels), 0)

      dev_loss = model.loss_criterion(out, log).item()
      dev_acc = compute_acc(log, out)
      eacc.append(acc)
      edev_loss.append(dev_loss)
      edev_acc.append(dev_acc) 

    band = False
    ep_finish_print = f' acc: {acc:.3f} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc:.3f}'

    if model.best_acc is None or model.best_acc < dev_acc:
      model.save(os.path.join(output, f'{model_name}_{split}.pt'))
      model.best_acc = dev_acc
      band = True

    if band == True:
      print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
    else: print(last_printed + ep_finish_print)
    
    with open('logs.txt', 'a') as file:
      file.write(f"\n {model_name}: {last_printed + ep_finish_print} \n")

  return {'loss': eloss, 'f1': eacc, 'dev_loss': edev_loss, 'dev_f1': edev_acc}


def train_model_CV(model_name, lang, data, splits = 5, epoches = 4, batch_size = 8, max_length = 120, 
                    interm_layer_size = 64, lr = 1e-5,  decay=2e-5, output='logs', model_mode='offline'):

  history = []
  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)
  
  model_params = {'mode':model_mode, 'lang':lang}
  for i, (train_index, test_index) in enumerate(skf.split(data['text'], data['humor'])):  
    
    history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
    model = SeqModel(interm_layer_size, max_length, **model_params)

    trainloader = DataLoader(Data({key: data[key][train_index] for key in data.keys()}), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
    devloader = DataLoader(Data({key: data[key][test_index] for key in data.keys()}), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)

    history.append(train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, i+1))
      
    print('Training Finished Split: {}'. format(i+1))
    del trainloader
    del model
    del devloader
  return history


def train_model_dev(model_name, lang, data_train, data_dev, epoches = 4, batch_size = 8, max_length = 120, 
                    interm_layer_size = 64, lr = 1e-5,  decay=2e-5, output='logs', model_mode='offline'):

  history = []
  
  model_params = {'mode':model_mode, 'lang':lang}
  history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
  model = SeqModel(interm_layer_size, max_length, **model_params)

  trainloader = DataLoader(Data(data_train), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
  devloader = DataLoader(Data(data_dev), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)

  history.append(train_model(model_name+lang, model, trainloader, devloader, epoches, lr, decay, output))

  del trainloader
  del model
  del devloader
  return history
