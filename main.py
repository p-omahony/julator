from utils import preprocessing

if __name__ == '__main__' :

    text_ds = preprocessing.build_input_ds(128, 20000, 80, ["./data/clean-jul-verses.txt"])
