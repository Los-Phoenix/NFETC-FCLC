# -------------------- PATH ---------------------
DATA_PATH = 'data'
WIKI_DATA_PATH = '%s/Wiki' % DATA_PATH
WIKIM_DATA_PATH = '%s/wikim' % DATA_PATH
ONTONOTES_DATA_PATH = '%s/OntoNotes' % DATA_PATH
BBN_DATA_PATH = '%s/BBN' % DATA_PATH

LOG_DIR = 'log'
CHECKPOINT_DIR = 'checkpoint'
OUTPUT_DIR = 'output'
PKL_DIR = 'pkl'
EMBEDDING_DATA = '%s/glove.840B.300d.txt' % DATA_PATH
testemb = '_emb'

# -------------------- DATA ----------------------

WIKI_ALL = '%s/all.txt' % WIKI_DATA_PATH
WIKI_TRAIN = '%s/train.txt' % WIKI_DATA_PATH
WIKI_VALID = '%s/dev.txt' % WIKI_DATA_PATH
WIKI_TEST = '%s/test.txt' % WIKI_DATA_PATH
WIKI_TRAIN_CLEAN = '%s/train_clean.tsv' % WIKI_DATA_PATH
WIKI_TEST_CLEAN = '%s/test_clean.tsv' % WIKI_DATA_PATH
WIKI_TYPE = '%s/type.pkl' % WIKI_DATA_PATH

WIKI_MAPPING = '%s/wiki_mapping.txt' % DATA_PATH

WIKIM_ALL = '%s/all.txt' % WIKIM_DATA_PATH
WIKIM_TRAIN = '%s/train.txt' % WIKIM_DATA_PATH
WIKIM_VALID = '%s/dev.txt' % WIKIM_DATA_PATH
WIKIM_TEST = '%s/test.txt' % WIKIM_DATA_PATH
WIKIM_TRAIN_CLEAN = '%s/train_clean.tsv' % WIKIM_DATA_PATH
WIKIM_TEST_CLEAN = '%s/test_clean.tsv' % WIKIM_DATA_PATH
WIKIM_GT_CLEAN = '%s/gt_clean.tsv' % WIKIM_DATA_PATH
WIKIM_TYPE = '%s/type.pkl' % WIKIM_DATA_PATH

ONTONOTES_ALL = '%s/all.txt' % ONTONOTES_DATA_PATH
ONTONOTES_TRAIN = '%s/train.txt' % ONTONOTES_DATA_PATH
ONTONOTES_VALID = '%s/dev.txt' % ONTONOTES_DATA_PATH
ONTONOTES_TEST = '%s/test.txt' % ONTONOTES_DATA_PATH
ONTONOTES_TRAIN_CLEAN = '%s/train_clean.tsv' % ONTONOTES_DATA_PATH
ONTONOTES_TEST_CLEAN = '%s/test_clean.tsv' % ONTONOTES_DATA_PATH
ONTONOTES_GT_CLEAN = '%s/gt_clean.tsv' % ONTONOTES_DATA_PATH
ONTONOTES_TYPE = '%s/type.pkl' % ONTONOTES_DATA_PATH

BBN_ALL = '%s/all.txt' % BBN_DATA_PATH
BBN_TRAIN = '%s/train.txt' % BBN_DATA_PATH
BBN_VALID = '%s/dev.txt' % BBN_DATA_PATH
BBN_TEST = '%s/test.txt' % BBN_DATA_PATH
BBN_TRAIN_CLEAN = '%s/train_clean.tsv' % BBN_DATA_PATH
BBN_TEST_CLEAN = '%s/test_clean.tsv' % BBN_DATA_PATH
BBN_TYPE = '%s/type.pkl' % BBN_DATA_PATH

# --------------------- PARAM -----------------------

MAX_DOCUMENT_LENGTH = 30

MENTION_SIZE = 15

WINDOW_SIZE = 10

RANDOM_SEED = 2017
