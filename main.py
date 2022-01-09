from utils import preprocessing

if __name__ == '__main__' :
    text = preprocessing.read_data("./data/jul-verses.txt")
    cleaned_lyrics = preprocessing.clean_lyrics(text)
    preprocessing.write_lyrics("./data/clean-jul-verses.txt", cleaned_lyrics)

    text = preprocessing.read_data("./data/naps-verses.txt")
    cleaned_lyrics = preprocessing.clean_lyrics(text)
    preprocessing.write_lyrics("./data/clean-naps-verses.txt", cleaned_lyrics)

    text_ds = preprocessing.build_input_ds(128, 20000, 80, ["./data/clean-jul-verses.txt", "./data/clean-naps-verses.txt"])
