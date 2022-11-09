import argparse, sys, os, numpy as np, torch, random

from utils.params import params, bcolors
from utils.utils import mergeData, TranslatePivotLang, backTranslation
from utils.utils import load_data, plot_training, evaluate
from models.SeqModels import train_model_CV, SeqModel, train_model_dev

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def check_params(args=None):
  
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-model', metavar='model', default = params.MODEL, 
      help='Model to be run')
  parser.add_argument('-mode', metavar='phase', default = params.MODE, 
      help='Train evaluate or encode with model', choices=['train', 'encode', 'predict', 'merge', 'translate'])
  parser.add_argument('-output', metavar='output', default = params.OUTPUT, 
      help='Output path for encodings and predictions')
  parser.add_argument('-lr', metavar='lrate', default = params.LR , type=float, 
      help='Learning rate for neural models optimization')
  parser.add_argument('-decay', metavar='decay', default = params.DECAY, type=float,
      help='learning rate decay for neural models optimization')
  parser.add_argument('-splits', metavar='splits', default = params.SPLITS, type=int, 
      help='spits cross validation on model training')
  parser.add_argument('-ml', metavar='max_length', default = params.ML, type=int,
       help='Maximun text length analysis')
  parser.add_argument('-wm', metavar='weigths_mode', default = params.PRET_MODE,
       help='Mode of pretraining weiths (online or offline) for transformers', choices=['online', 'offline'])
  parser.add_argument('-interm_layer', metavar='int_layer', default = params.IL, type=int,
       help='amount of intermediate layer neurons')
  parser.add_argument('-epoches', metavar='epoches', default=params.EPOCHES, type=int,
       help='Trainning epoches')
  parser.add_argument('-bs', metavar='batch_size', default=params.BS, type=int,
       help='Batch Size')
  parser.add_argument('-l', metavar='lang',
       help='Language')
  parser.add_argument('-tf', metavar='train_file',
       help='Data anotation files for training')
  parser.add_argument('-df', metavar='test_file', 
       help='Data anotation files for testing')
  parser.add_argument('-wp', metavar='weigths_path', default="logs",
       help='Saved weights Path')

  return parser.parse_args(args)


if __name__ == '__main__':


  parameters = check_params(sys.argv[1:])

  model = parameters.model
  mode = parameters.mode
  output = parameters.output
  
  learning_rate, decay = parameters.lr,  parameters.decay
  splits = parameters.splits
  max_length = parameters.ml

  weights_mode = parameters.wm
  interm_layer_size = parameters.interm_layer
  epoches = parameters.epoches
  batch_size = parameters.bs
  lang = parameters.l  

  tf = parameters.tf
  df=parameters.df

  weights_path = parameters.wp
  

  if model == 'encoder':

    if mode == 'train':

      output = os.path.join(output, 'logs')

      if os.path.exists(output) == False:
        os.system(f'mkdir {output}')

      dataTrain = load_data(tf, lang)
      history = None
      
      if df is None:
        history = train_model_CV(model_name=params.models[lang].split('/')[-1], lang=lang, data=dataTrain, splits=splits, epoches=epoches, 
                      batch_size=batch_size, max_length=max_length, interm_layer_size = interm_layer_size, 
                      lr = learning_rate,  decay=decay, output=output, model_mode=weights_mode)
      else:
        dataDev = load_data(df, lang)
        history = train_model_dev(model_name=params.models[lang].split('/')[-1], lang=lang, data_train=dataTrain, data_dev=dataDev,
                      epoches=epoches, batch_size=batch_size, max_length=max_length, 
                      interm_layer_size = interm_layer_size, lr = learning_rate,  decay=decay, 
                      output=output, model_mode=weights_mode)
      
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Training Finished for {lang.upper()} Model{bcolors.ENDC}")
      plot_training(history[-1], f'lm_{lang}', 'loss')
      exit(0)

    if mode in ['encode', 'predict']:

      dataDev = load_data(df, lang)
      mode = 'encodings' if mode == 'encode' else 'preds'

      outputFile = df.split('/')[-1].split('.')[0]
      outputFile = os.path.join(output, outputFile) + f'_{lang}_{mode}.csv'

      model_params = {'mode':weights_mode, 'lang':lang}
      model = SeqModel(interm_layer_size, max_length, **model_params)
      model.evaluate(data=dataDev, batch_size=batch_size, wp= weights_path, outputFile = outputFile, encode=(mode == 'encodings'))
      
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}{mode.title()} Saved{bcolors.ENDC}")
    exit(0)

  if model == 'makedata':

    if mode == 'merge':
      mergeData('train')
      mergeData('test')
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Dataset Made!!{bcolors.ENDC}")
    else:
      #TranslatePivotLang(tf, output)
      backTranslation(tf)

      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Data Translated!!{bcolors.ENDC}")
    exit(0)

  if model == 'eval':
    evaluate(df)
    exit(0)
