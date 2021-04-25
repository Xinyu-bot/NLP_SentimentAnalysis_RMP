### Abstract
__...__

### Project Intro
__...__

### Dataset
#### Lexicon
* __HLTEMNLP__: Theresa Wilson, Janyce Wiebe and Paul Hoffmann (2005). Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis. Proceedings of HLT/EMNLP 2005, Vancouver, Canada.
* __Vader__: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
* __SentiWords__ (_Disqualified due to bad labeling_): Gatti, Lorenzo, Marco Guerini, and Marco Turchi. "SentiWords: Deriving a high precision and high coverage lexicon for sentiment analysis." IEEE Transactions on Affective Computing 7.4 (2016): 409-421. 

#### Corpus
* __IMDB__: https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format
* __RMP__: retrieved from RateMyProfessor.com and labeled ourselves
* __twitter1600k__: shit, no longer used
* __More...__


### Methods 
* lexicon-based unigram model
* corpus-based bigram model (with unigram back-off)
* corpus-based trigram model (with bigram model back-off)
* vector similarity
* 然后如果有兴趣的话可以加一个deep learning with tensorflow


### Result
_will rerun each test after RMP corpus is sufficiently large_
* lexicon-based 
* bigram
* trigram
* vector similarity
* __determine which method is the best__

### Conclusion and Future/Further Work
* negation
* tensorflow / deep learning
* quad-gram
* larger and more accurately labeled corpus