### Abstract
__...__

### Project Intro
__...__

### Dataset
#### Lexicon
* __HLTEMNLP__: 
  * Lexicon of polarity of around 8k words. Categorical values only: positive, neutral, negative
  * _Theresa Wilson, Janyce Wiebe and Paul Hoffmann (2005). Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis. Proceedings of HLT/EMNLP 2005, Vancouver, Canada._

* __Vader__: 
  * Lexicon of polarity of around 7k words. Decimal values based on average of 10 human judges' ratings on the word: scale from -4 to 4. 
  * _Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014._

* __SentiWords__ (_Disqualified due to bad labeling_): 
  * Lexicon of polarity of around 155k words. Decimal values scale from -1 to 1. In our opinion. this lexicon is badly labeled as it contains contradicting labellings on different morphologies of a same word. And in fact, adding this lexicon to our system significantly decreases the performance in every aspect. 
  * _Gatti, Lorenzo, Marco Guerini, and Marco Turchi. "SentiWords: Deriving a high precision and high coverage lexicon for sentiment analysis." IEEE Transactions on Affective Computing 7.4 (2016): 409-421._

#### Corpus
* __IMDB Film Reviews__: 
  * Retrieved from https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format 
  * 25k reviews labeled as positive, another 25k as negative. 
* __RMP__: 
  * Retrieved from RateMyProfessor.com and labeled ourselves
  * Total count is increasing, as for 4.27, we have over 30k comments labeled as positive, and over 9k comments labeled as negative. Will catch up the amount of negative comments in the future. 
* __Twitter1600k__: 
  * retrieve online but already forgot the exact source. 
  * contains tons of slangs, illegal grammar/spelling/punctuation. 
  * adding this corpus to our system will significantly lowers performance in every aspect. 
  * will not be used regardless, so forgetting about the source is OK. 
* __Maybe More...__: 
  * we might add more corpus to train our system in order to increase the generalizability of our system, 
  * i.e. make our system useable for not only sentiment analysis on RMP, but also on other commentary dataset... 


### Methods 
* __lexicon-based unigram model__
  * build unigram model based on the lexicon we have, and examine each sentence unigram by unigram. 
* __corpus-based unigram model__
  * build unigram model based on the corpus we have, and examine each sentence unigram by unigram. 
* __corpus-based bigram model__ (with unigram back-off)
  * build a bigram model and a unigram model based on the corpus we have, and examine each sentence bigram by bigram
  * if a specific bigram sequence is not found in our bigram model, we back-off to the unigram model. 
* __corpus-based trigram model__ (with bigram model back-off)
  * same as bigram model above
  * but we back-off to bigram model first, and if still not found, we back-off to unigram model
* __vector similarity__
  * TF-IDF score for vectorized sentences. 
  * compute vector similarity between each input sentence and the corpus, find the top 20 most related instances in the corpus and compute average sentiment of the instances. 

### Result
We shuffle our corpus first to add more randomization into the sampling process. Then we train our system with 70% of the corpus, test with the 30% left, then record the Precisions, Recalls, and F-measures on Positive, Negative, and Overall. Data shown below is only for reference and should not be considered _finalized_. 

**_After RMP corpus is sufficiently large in the end and all methods are optimized and finalized, we will rerun each test and use that results for our paper._**

* __lexicon-based unigram__
  * middle in the 70s
* __corpus-based unigram__ 
  * middle in the 60s
* __bigram__
  * middle in the 80s 
* __trigram__
  * high in the 80s
* __vector similarity__
  * around 80s
* __Performance Analysis on the Statistics__ 
  * Right now, Trigram system has the best performance in terms of speed and the statistics. 
  * In fact, it is designed to be the main / core method of our sentiment analysis system, and it will be so for the finalized version of our system. 

### Conclusion and Future/Further Work
* __Conclusion__ 
  * No conclusion is drawn so far since our project is not finalized yet. 

* __Future/Further Work__
  * Special handling on negation and hyperbaton
  * `TensorFlow.Keras` feat. `Word2Vec`; or, more generally, deep learning
  * Quad-gram... (is this really worth doing? )
  * Larger and more accurately labeled corpus