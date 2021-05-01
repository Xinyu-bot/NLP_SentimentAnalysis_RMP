import trigram
import scraper
import unigram_lexicon_based
from nltk import PorterStemmer
from time import time
from nltk.corpus import stopwords


# main function of the application
def main() -> None: 
    # read user's input
    mode = int(input("Do you want to search directly by URL of professor on RateMyProfessor website or by professor name:\n\t0 for URL, 1 for name.\n "))
    while mode != 0 and mode != 1: 
        mode = int(input("Invalid entry. Please enter 0 for URL, or 1 for name. "))
    print("...")
    key = input("Please enter the content you want to search for, either a name or a URL.\n ")
    print("...")

    # JIT data retrival and unpacking from RMP 
    try:
        comments, quality_score, difficulty_score, name = scraper.get_comments(key, mode)
    except AssertionError: 
        print("[ERROR] Invalid URL. Please check its correctness and try again. ")
        return
    print("...")

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


    pos, neg = 0, 0
    weight = [0, 0]
    count = 0
    # loop through the comments
    for comment in comments: 
        res = trigram.analyze_trigram(comment, trigram_model, bigram_model, unigram_model, STOP_WORDS, porterStemmer)
        _pos, _neg = res
        if _pos > _neg: 
            pos += 1
        elif _neg > _pos: 
            neg += 1
        else: 
            pass
        weight[0] += _pos
        weight[1] += _neg
        count += 1
    
    sentiment_score = round(3.0 + (4.0 * ((pos / (pos + neg)) - 0.5)), 1)
    sentiment_score2 = round(3.0 + (4.0 * ((weight[0] / count) - 0.5)), 1)

    print("=" * 42)
    print("AHSAR result on professor {0}:\nQuality Score: \t\t\t{1}\nDifficulty Score: \t\t{2}\nSentiment Score (discrete): \t{3}\nSentiment Score (continuous): \t{4}"
            .format(name, quality_score, difficulty_score, sentiment_score, sentiment_score2)
            )
    print("\nNotice that discrete sentiment score is computed based on individual comments, while continuous sentiment score is computed based on all comments. ")
    print("In other words, the higher the discrete score is, the more individual comments are positive. The higher the continuous score is, the larger proportion of all comments are positive. ")
    print("=" * 42)

    return 

# always a good practice
if __name__ == '__main__': 
    print('Welcome. ')

    flag = 'y'
    while flag == 'y': 
        main()
        flag = input("Continue to next professor? [y/n] ")
        while flag != 'y' and flag != 'n': 
            flag = input("Invalid entry. Enter again... [y/n] ")

    print("Bye. ")
