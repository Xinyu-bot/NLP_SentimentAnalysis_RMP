from unigram import *
from nltk.tokenize import word_tokenize, sent_tokenize
from scraper import main as grab

def comment_parsing(comment: str, lexicon: Lexicon) -> dict: 

    comment_sentiment = []
    sentences = sent_tokenize(comment)
    # loop through each sentence
    for sentence in sentences: 
        sentence_sentiment = []
        clauses = sentence.split(',')
        for clause in clauses: 
            flag = False
            clause = word_tokenize(clause)
            #print(' '.join(clause))
            # loop through the clause
            for token in clause: 
                token = token.lower()
                # try to get token info from lexicon
                if token == 'not' or token == "n't": 
                    flag = True
                if token == 'but': 
                    flag = False

                try: 
                    word_obj = lexicon._get(token)
                    #print(word_obj.word)
                    _word, _pos, _sentiment = word_obj.word, word_obj.pos, word_obj.sentiment
                    
                    if flag: 
                        if _sentiment == 'positive':
                            _sentiment = 'negative'
                        elif _sentiment == 'negative': 
                            _sentiment = 'positive'
                    
                    sentence_sentiment.append(_sentiment)
                # if not in lexicon, ignore
                except (AttributeError):
                    pass 

        #print(sentence_sentiment)
        comment_sentiment += sentence_sentiment
    
    return comment_sentiment

def sentiment_analysis(comment_sentiment: list) -> None:
    #print(comment_sentiment)
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    for element in comment_sentiment: 
        if element == 'positive': 
            positive_count += 1
        elif element == 'negative': 
            negative_count += 1
        else: 
            neutral_count += 1

    #print(positive_count, negative_count, neutral_count)
    count_sum = positive_count * 5 + negative_count * 5 + neutral_count
    try: 
        weight = {
            'positive': round(positive_count * 5 / count_sum, 3), 
            'negative': round(negative_count * 5 / count_sum, 3), 
            'neutral':  round(neutral_count / count_sum, 3)
        }
    except ZeroDivisionError: 
        weight = {
            'positive': 0, 
            'negative': 0, 
            'neutral':  0
        }

    print("This comment has weighed sentiment as: \n\tpositive: {0},\
            \n\tnegative: {1}, \n\tneutral: {2}"
            .format(weight['positive'], weight['negative'], weight['neutral']))


if __name__ == '__main__': 

    #prof_list = grab()
    #comment = prof_list[0]['comments'][0][2]

    comment = "His lectures are painful; he reads dense powerpoints with lots of jargon, and it's unclear what's important and what's filler. The assignments are difficult, but more because there's little to no guidance given. The tests are much easier, so this is a relatively easy class but I'd look elsewhere."

    lexicon = generate_lexicon()
    comment_sentiment = comment_parsing(comment, lexicon)
    sentiment_analysis(comment_sentiment)