from utils import preprocessing
from utils.preparation import WordSeqDataLoader
from models.lstm import lstm_data_prepare
import numpy as np

if __name__ == '__main__' :
    # clean_jul = preprocessing.CleanData('./data/jul-verses.txt')
    # clean_jul('./data/clean-jul-verses.txt')
    #
    # clean_naps = preprocessing.CleanData('./data/naps-verses.txt')
    # clean_naps('./data/clean-naps-verses.txt')

    corpus = open('./data/clean-jul-verses.txt', 'r').read()
