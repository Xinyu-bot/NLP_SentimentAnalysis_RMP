import os
from nltk.tokenize import word_tokenize, sent_tokenize

''' 
a = Lexicon()
word = 'hello'
sentiment = 'negative'
setattr(a, word, Token(word, 'verb', sentiment))
print(a.hello.word) --> 'hello'
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
    __slots__ = ('word', 'sentiment')
    def __init__(self, word, sentiment):
        self.word = word
        self.sentiment = sentiment


# reading in the file
def generate_lexicon(filename: str) -> Lexicon: 
    # create the lexicon class
    unigram = Lexicon()

    # read in the file
    with open(filename, 'r') as instream: 
        for line in instream: 
            line = line.strip(os.linesep).split(' ')
            word, sentiment = line[2].split('=')[1], line[5].split('=')[1]
            setattr(unigram, word, Token(word, sentiment))

    return unigram


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
                    _word, _sentiment = word_obj.word, word_obj.sentiment
                    
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
    
    # print(comment_sentiment, '\n')
    return comment_sentiment

def sentiment_analysis(comment_sentiment: list) -> tuple:
    #print(comment_sentiment)
    positive_count = 0
    negative_count = 0
    for element in comment_sentiment: 
        # if element > 0: 
        if element == 'positive': 
            positive_count += 1
        # elif element < 0: 
        elif element == 'negative': 
            negative_count += 1
        else: 
            continue

    #print(positive_count, negative_count, neutral_count)
    count_sum = positive_count + negative_count
    try: 
        # (positive, negative)
        weight = (round(positive_count / count_sum, 3), round(negative_count / count_sum, 3))
    except ZeroDivisionError: 
        weight = (0, 0)

    # print("This comment has weighed sentiment as: \n\tpositive: {0}, negative: {1}"
    #        .format(weight[0], weight[1]))

    # print(weight, '\n')
    return tuple(weight)

''' 
if __name__ == '__main__':
    comments = [
        "Adam Meyers is an idiosyncratic individual. Has very dense lectures, but I find them interesting. Obviously not everyone will. Helps to have some prior knowledge of programming. Very knowledgeable and approachable (important for big lectures). Overall great guy with stellar taste in music and fashion.", 
        "Grading was very fair, mostly depend on your understanding of the knowledge instead of how hardworking you seem to be. I barely went to any class bc I was in an 8am section. Attendance was not calculated into the grade. All lectures + modules are accessible on the class website. I understood everything and did all hws+tests well so it was an easy A", 
        "Took his undergrad NLP class. His lectures can be dense (and yes, a little boring), so if you have trouble paying attention to his slides, going to office hours is KEY. He is extremely intelligent in this field, and above all else he wants his students to gain something from the class. The labs are all doable and relevant to the material. 10/10.", 
        "He is probably the best professor out there in NYU. Really lenient on grading criteria and you just have to go to every class and do the problem sets and you will do well. Quizzes get dropped, problem sets get dropped and even one of the two midterms too if you do bad on one. Finals matter the most.", 
        "Very good professor, focuses on learning rather than memorizing, lenient grader. Homeworks are not reaally easy but doable and not that long. All the help is provided in the prompt such as how to approach the problem.", 
        "Having his background in linguistics, he is keen on making python feel useful and accessible to non-stem or non-comp sci majors. Really nice guy. Lectures can get dull sometimes, but he tends to use good examples. Average difficulty homework and exams. The general feel of the class was very laid back. Would take him again!",             
        "Despite the general difficulty of the content, I've learned more in his class than I've learned in other economics classes. As there's no standardized curriculum for this class across the department, he chooses particularly relevant and fundamental topics that have helped me in every other economics class that I've taken. Professor Madsen rocks.", 
        "The best math professor at nyu, no doubt. His lectures are clear and easy to follow along. He doesn't try and trick you on the exams as you are literally tested on what he teaches you from lectures. Homework was a bit more challenging but doable. I took him over the summer, so he taught material faster. If you have the chance to take him, do it!",

        "He teaches so fast that I don't even know what he is saying most of the times. His demonstrations in class are also very unclear because it is on things that we haven't even learned. He literally expects us to know python when this class is for beginners.", 
        "This should be a no prior computer science experience foundation course, but he teaches the course so fast that students without any computer science experience find it difficult to catch with his progress. On the lectures he does programming demonstrations that students even don't have any foundations on.", 
        "Adam is very kind as a person and is easy to approach and talk to. However, as a professor, his teaching style results in extremely tedious and dull lectures that ultimately have turned me away from the subject matter. His lectures are unbelievably monotonous and hard to follow. Homeworks are long and tedious. Exams are a breeze.", 
        "His lectures are painful; he reads dense powerpoints with lots of jargon, and it's unclear what's important and what's filler. The assignments are difficult, but more because there's little to no guidance given. The tests are much easier, so this is a relatively easy class but I'd look elsewhere."
        "Firstly, Madsen is very knowledgable but he tries to prove this. We were given an Intermediate Micro textbook, however we don't follow it and go far beyond the class. Most lectures do not cover the economics topic but dive into multivariable calc with lots of confusing notation. Feel like I can do math but don't actually understand the economics.", 
        "Short and Simple: Do not take if you care about your GPA. Madsen is not a good professor, and his tests are beyond what is learned in class or homework's.", 
        "This guy does not know the first thing about writing exams. He asked students in his Intermediate Microeconomics to build an ALGORITHM on the exam. Who does that? He is a terrible professor and he does not have the slightest clue about the actual caliber of undergraduate students and thinks we're all PhD students back in Stanford with him.", 
        "Professor Madsen is going to ruin your GPA and you are not going to learn. His homework are extremely difficult and his exams are even more exponentially difficult, impossible even. His exams and homework do not resemble the slides he teaches from, which he did not even make. He is the worst professor I've ever taken in my life."
        ]

    infile_name = '../data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'


    lexicon = generate_lexicon(infile_name)

    for comment in comments: 
        comment_sentiment = comment_parsing(comment, lexicon)
        sentiment_analysis(comment_sentiment)
''' 