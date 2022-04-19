
class params:

  models = {'fr': 'flaubert/flaubert_base_cased', 'en': 'vinai/bertweet-base',
            'es':'finiteautomata/beto-sentiment-analysis', 'de':'oliverguhr/german-sentiment-bert',
            'it': 'dbmdz/bert-base-italian-uncased', 'pt':'neuralmind/bert-base-portuguese-cased', 'ml':'bert-base-multilingual-uncased-sentiment'}
  
  LR, DECAY = 1e-5,  2e-5
  SPLITS = 5
  IL = 64
  ML = 110
  BS = 64
  EPOCHES = 4
  MULTITASK = 'stl'
  PRET_MODE = 'offline'
  OUTPUT = '.'
  MODEL = 'encoder'
  MODE = 'train'
  
class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


