import os

''' 
a = Lexicon()
word = 'hello'
sentiment = 'negative'

setattr(a, word, Token(word, 'verb', sentiment))

print(a.hello.pos) --> verb

Lexicon: 
self.pervert -> self.pervert.word
self.phobic
self.phony
'''
class Lexicon: 
    __slot__ = ('__dict__')
    def __init__(self): 
        pass

    def _get(self, token):
        return object.__getattribute__(self, token)

class Token: 
    __slots__ = ('word', 'pos', 'sentiment')
    def __init__(self, word, pos, sentiment):
        self.word = word
        self.pos = pos
        self.sentiment = sentiment


# reading in the file
def generate_lexicon() -> Lexicon: 
    # create the lexicon class
    unigram = Lexicon()

    # read in the file
    with open('../data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff', 'r') as instream: 
        for line in instream: 
            line = line.strip(os.linesep).split(' ')
            word, pos, sentiment = line[2].split('=')[1], line[3].split('=')[1], line[5].split('=')[1]
            setattr(unigram, word, Token(word, pos, sentiment))

    return unigram

if __name__ == '__main__':
    unigram = generate_lexicon() 
    print(unigram.phenomenal.sentiment)