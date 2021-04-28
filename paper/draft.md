## Group 8 Sentiment Analysis on RateMyProfessor
Xinyu Xie, Qian (Chasity) Chen, Yutang (Tony) Li

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
* __Twitter1600k__  (_Disqualified due to bad data content_): 
  * Retrieve online but already forgot the exact source. 
  * Contains tons of slangs, illegal grammar/spelling/punctuation. 
  * Adding this corpus to our system will significantly lowers performance in every aspect. 
  * Will not be used regardless, so forgetting about the source is OK. 
* __Maybe More...__: 
  * We might add more corpus to train our system in order to increase the generalizability of our system, 
  * i.e. make our system useable for not only sentiment analysis on RMP, but also on other commentary dataset... 


### Methods 
#### Self-implemented, meaning 100% originality and built from scratch
* __lexicon-based unigram model__
  * Build unigram model based on the lexicon we have, and examine each sentence unigram by unigram. 
* __corpus-based unigram model__
  * Build unigram model based on the corpus we have, and examine each sentence unigram by unigram. 
* __corpus-based bigram model__ (with lexicon-based unigram model back-off)
  * Build a bigram model and a unigram model based on the corpus we have, and examine each sentence bigram by bigram
  * If a specific bigram sequence is not found in our bigram model, we back-off to the unigram model. 
* __corpus-based trigram model__ (with bigram model back-off)
  * Same as bigram model above
  * But we back-off to bigram model first, and if still not found, we back-off to unigram model
* __vector similarity model__
  * Re-use code written for the IR homework. No `sklearn` module involved.
  * TF-IDF score for vectorized sentences then compute vector similarity
  * Use average sentiment score for top N instances from the corpus

#### Localization based on Others systems
_Tons of library / module usage_
* __Count Vector and TFIDF Vector__
  * On _Linear Regression_, _Stochastic Gradient Descent_, and _Multinomial Naive Bayes_ classification models
  * Rework based on our corpus and needs

* __Word2Vector__
  * Simply speaking, just another TFIDF vector
  * Involves `TensorFlow.Keras`'s `Sequential` model
  * Rework based on our corpus and needs
  
### Result
We shuffle our corpus first to add more randomization into the sampling process. Then we train our system with 70% of the corpus, test with the 30% left, then record the Precisions, Recalls, and F-measures on Positive, Negative, and Overall. Data shown below is only for reference and should not be considered _finalized_. 

**_After RMP corpus is sufficiently large in the end and all methods are optimized and finalized, we will rerun each test and use that results for our paper._**
#### Statistics
* __Lexicon-based Unigram__
  * Variance very large; very bad at negative sentiment detection
  * Average at middle in the 60s
* __Corpus-based Unigram__ 
  * Same with Lexicon-based Unigram
  * Middle in the 60s
* __Bigram__
  * Embedded in the Trigram system, actually
  * Very fast speed after optimization
  * Few measures hit 90, and average at almost 90s 
* __Trigram__
  * Very fast speed after optimization
  * High in the 80s, almost hit 90 but about 1% off from bigrams
* __Vector Similarity__
  * Very badly optimized code in terms of running speed
  * Around 80s
* __Count Vector and TFIDF Vector__
  * Linear Regression: 
    * Around 70s for both CV and TV
  * Stochastic Gradient Descent: 
    * Around 50s for both CV and TV
  * Multinomial Naive Bayes: 
    * Around 70s for both CV and TV
  * Numbers are astonishingly low, but the original author got similar result on his corpus too
  * Definitely incomparable to our simple-to-build, simple-to-train, simple-to-use model
* __Word2Vector__ 
  * Dimension of vector for the neural network is set up to `500`, but not too helpful
  * Middle in the 80s
  * Might need future work on this to improve its score
  * But being 5% or so less than our model probably means it does not worths to be improved
#### __Performance Analysis__ 
  * Right now, our own N-gram system has the best performance over Vector Similarity system, and the other two Vector-related systems in terms of speed and statistics. 
  * In fact, it is designed to be the main / core method of our sentiment analysis system from the beginning, and it will be so for the finalized version of our system. 
  * We've read a few online blogs about using a `Sequential` model of `TensorFlow.Keras` and incorporating other modules like `word2vec` and `sklearn`. Statistics for such a simply/naively implemented neural network are about high 70s or low 80s, which is significantly lower than our N-grams system. Also, our N-grams model takes much less time to run. 

### Conclusion and Future/Further Work
* __Conclusion__ 
  * The myth of **_"Power of Two"_** confirmed! We achieve huge improvement from unigram to bigram, but less (well, actually none) improvement from bigram to trigram...
  * So far, our model has loosely proved its performance against some models available online
  * No other conclusion is drawn so far since our project is not finalized yet. 

* __Future/Further Work__
  * _Will do_: 
    * Error analysis and optimization on performance
    * Larger and more accurately labeled corpus
  * _No guarantee_: 
    * Special handling on negation and hyperbaton: not, never, but, however, etc.
    * Detection of Sarcasm: we read blogs and essays that this is one of the frontier topics
    * Sentence or clause boundary preservation for Trigram 
    * Extension on N-gram: maybe quad-gram... (is this really worth doing? Trigram's negative improvement is a thing)
  
