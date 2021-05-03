import trigram
import scraper
import unigram_lexicon_based
from nltk import PorterStemmer
from time import time
from nltk.corpus import stopwords

''' Modulability is everything '''
def load_models() -> tuple: 
    # import models and setup for sentiment analysis process
    a = time()
    unigram_file_extended = '../data/unigram/unigram_lexicon_extended.csv'
    unigram_model = unigram_lexicon_based.generate_lexicon(unigram_file_extended)
    trigram_model, bigram_model, _ = trigram.import_models()
    porterStemmer = PorterStemmer()
    STOP_WORDS = []
    # STOP_WORDS += stopwords.words('english')
    b = time()
    print("Models loaded in {0} sec. ".format(round((b - a), 2)))

    return trigram_model, bigram_model, unigram_model, porterStemmer, STOP_WORDS

def fetch_prof_info() -> tuple: 
    # ask for name or URL
    try:  
        mode = int(input("Do you want to search directly by URL of professor on RateMyProfessor website or by professor name:\n0 for URL, 1 for name.\n "))
        while mode != 0 and mode != 1: 
            mode = int(input("Invalid entry. Please enter 0 for URL, or 1 for name. "))
    except ValueError: 
        print("[ERROR] Invalid Entry. Please check the requirement and try again. ")
        raise AssertionError

    print("...")

    key = input("Please enter the content you want to search for, either a name or a URL.\n ")
    print("...")

    # JIT data retrival and unpacking from RMP 
    try:
        comments, quality_score, difficulty_score, name = scraper.get_comments(key, mode)
    except AssertionError: 
        print("[ERROR] Invalid URL. Please check its correctness and try again. ")
        raise AssertionError

    print("...")

    return comments, quality_score, difficulty_score, name

def analyze_sentiment(comments: list, trigram_model: dict, bigram_model: dict, unigram_model: dict, STOP_WORDS: list, 
             porterStemmer: PorterStemmer, quality_score: float, difficulty_score: float, name: str) -> None: 
    pos, neg, count = 0, 0, 0
    weight = [0, 0]
    # loop through the comments
    for comment in comments: 
        # update the counts
        _pos, _neg = trigram.analyze_trigram(comment, trigram_model, bigram_model, unigram_model, STOP_WORDS, porterStemmer)
        if _pos > _neg: 
            pos += 1
        elif _neg > _pos: 
            neg += 1
        else: 
            pass
        weight[0] += _pos
        weight[1] += _neg
        count += 1
    
    # compute two sentiment scores
    sentiment_score = round(3.0 + (4.0 * ((pos / (pos + neg)) - 0.5)), 1)
    sentiment_score2 = round(3.0 + (4.0 * ((weight[0] / count) - 0.5)), 1)

    # display the results
    print("=" * 42)
    print(
        "AH-SAR result on professor {0}:\nQuality Score: \t\t\t{1}\nDifficulty Score: \t\t{2}\nSentiment Score (discrete): \t{3}\nSentiment Score (continuous): \t{4}"
        .format(name, quality_score, difficulty_score, sentiment_score, sentiment_score2)
    )
    print("=" * 42)

    return

''' main function '''
# main function of the application
def main() -> None: 
    # load models
    trigram_model, bigram_model, unigram_model, porterStemmer, STOP_WORDS = load_models()

    # instruction info
    print("Notice that Sentiment Score (discrete) is computed based on individual comments, \nwhile Sentiment Score (continuous) is computed based on all comments. ")
    print("In other words, the higher the discrete score is, the more individual comments are positive. \nThe higher the continuous score is, the larger proportion of all comments are positive. \n")

    # read user's input
    flag = 'y'
    while flag == 'y': 
        # actual analysis 
        try: 
            # get professor info
            comments, quality_score, difficulty_score, name = fetch_prof_info()
            # check info validity
            if comments is None: 
                print("Professor {0} does not have any comment. ".format(name))
                pass
            # sentiment analysis
            analyze_sentiment(comments, trigram_model, bigram_model, unigram_model, STOP_WORDS, porterStemmer, quality_score, difficulty_score, name)
        except (AssertionError, TypeError) as e:
            pass

        # ask user for next round
        flag = input("Continue to next professor? [y/n] ")
        while flag != 'y' and flag != 'n': 
            flag = input("Invalid entry. Enter again... [y/n] ")

    return 

# always a good practice
if __name__ == '__main__': 
    print('Welcome. ')
    main()
    print("Bye. ")
