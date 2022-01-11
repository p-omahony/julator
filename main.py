from utils import preprocessing

if __name__ == '__main__' :
    clean_jul = preprocessing.CleanData('./data/jul-verses.txt')
    clean_jul('./data/clean-jul-verses.txt')

    clean_naps = preprocessing.CleanData('./data/naps-verses.txt')
    clean_naps('./data/clean-naps-verses.txt')

    # text = preprocessing.read_data("./data/jul-verses.txt")
    # cleaned_lyrics = preprocessing.clean_lyrics(text)
    # preprocessing.write_lyrics("./data/clean-jul-verses.txt", cleaned_lyrics)
    #
    # text = preprocessing.read_data("./data/naps-verses.txt")
    # cleaned_lyrics = preprocessing.clean_lyrics(text)
    # preprocessing.write_lyrics("./data/clean-naps-verses.txt", cleaned_lyrics)
