class WordSeqDataLoader :
    def __init__(self, corpus, word=True):
        self.corpus = corpus.replace('\n', ' ')
        self.word = word

        if word:
            self.vocab = sorted(list(set(corpus.split())))
            self.corpus_size = len(corpus.split())
        else:
            self.vocab = sorted(list(set(corpus)))
            self.corpus_size = len(corpus)
        self.vocab_size = len(self.vocab)
        self.v_to_int = dict((c, i) for i, c in enumerate(self.vocab))
        self.int_to_v = dict((i, c) for i, c in enumerate(self.vocab))

    def to_seq(self, seq_len):
        dataX, dataY = [], []
        for i in range(0, self.corpus_size - seq_len, 1):

            if self.word:
            	seq_in = self.corpus.split()[i:i + seq_len]
            	seq_out = self.corpus.split()[i + seq_len]
            else:
                seq_in = self.corpus[i:i + seq_len]
                seq_out = self.corpus[i + seq_len]

            dataX.append([self.v_to_int[e] for e in seq_in])
            dataY.append(self.v_to_int[seq_out])

        n_patterns = len(dataX)
        return dataX, dataY, n_patterns
