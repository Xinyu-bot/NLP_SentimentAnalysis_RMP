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

    comments = [
            "Adam Meyers is an idiosyncratic individual. Has very dense lectures, but I find them interesting. Obviously not everyone will. Helps to have some prior knowledge of programming. Very knowledgeable and approachable (important for big lectures). Overall great guy with stellar taste in music and fashion.", 
            "Grading was very fair, mostly depend on your understanding of the knowledge instead of how hardworking you seem to be. I barely went to any class bc I was in an 8am section. Attendance was not calculated into the grade. All lectures + modules are accessible on the class website. I understood everything and did all hws+tests well so it was an easy A", 
            "Took his undergrad NLP class. His lectures can be dense (and yes, a little boring), so if you have trouble paying attention to his slides, going to office hours is KEY. He is extremely intelligent in this field, and above all else he wants his students to gain something from the class. The labs are all doable and relevant to the material. 10/10.", 
            "He is probably the best professor out there in NYU. Really lenient on grading criteria and you just have to go to every class and do the problem sets and you will do well. Quizzes get dropped, problem sets get dropped and even one of the two midterms too if you do bad on one. Finals matter the most.", 
            "Very good professor, focuses on learning rather than memorizing, lenient grader. Homeworks are not reaally easy but doable and not that long. All the help is provided in the prompt such as how to approach the problem.", 
            "Having his background in linguistics, he is keen on making python feel useful and accessible to non-stem or non-comp sci majors. Really nice guy. Lectures can get dull sometimes, but he tends to use good examples. Average difficulty homework and exams. The general feel of the class was very laid back. Would take him again!",             

            "He teaches so fast that I don't even know what he is saying most of the times. His demonstrations in class are also very unclear because it is on things that we haven't even learned. He literally expects us to know python when this class is for beginners.", 
            "This should be a no prior computer science experience foundation course, but he teaches the course so fast that students without any computer science experience find it difficult to catch with his progress. On the lectures he does programming demonstrations that students even don't have any foundations on.", 
            "Adam is very kind as a person and is easy to approach and talk to. However, as a professor, his teaching style results in extremely tedious and dull lectures that ultimately have turned me away from the subject matter. His lectures are unbelievably monotonous and hard to follow. Homeworks are long and tedious. Exams are a breeze.", 
            "His lectures are painful; he reads dense powerpoints with lots of jargon, and it's unclear what's important and what's filler. The assignments are difficult, but more because there's little to no guidance given. The tests are much easier, so this is a relatively easy class but I'd look elsewhere."
            ]

    lexicon = generate_lexicon()

    for comment in comments: 
        comment_sentiment = comment_parsing(comment, lexicon)
        sentiment_analysis(comment_sentiment)