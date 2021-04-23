from nltk import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
from time import time
import math
import pandas as pd
import os

closed_class_stop_words = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever', 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ] 

# override to empty because the words above are not good enough
closed_class_stop_words = []
# extend stop_words List by nltk stop words
closed_class_stop_words += stopwords.words("english")                     

# helper function removing undesired tokens and return the cleanned tokens for modulability 
def tokens_cleaner(tokens: list) -> set:
    # evil set subtraction for best performance yet no existing order preserved
    output = {token.strip('\'"-.') for token in (set(tokens) - set(closed_class_stop_words) - \
            set(" ,./><?;':\"][\}{|~`!@#$%^&*)(_+-=")) if len(token) > 1}
    # (deprecated) return a subset of previous cleaned results by furthur cleaning 
    return output

# helper function computing IDF and update Dictionary accordingly for modulability 
def IDF_updater(docCount: int, tokenOccurrence: dict, tokenIDF: dict) -> None:
    # loop through `tokenOccurrence`
    for key, value in tokenOccurrence.items():
        # compute IDF for each token and update tokenIDF
        IDF = math.log(docCount / value)
        tokenIDF[key] = IDF

# helper function computing TFIDF and update Dictionary accordingly for modulability 
def TFIDF_updater(element: str, tokens: str, featureVector: list, index: int, tokenIDF: dict) -> None:
    # normalize occurrence of specific token by: 
    # dividing it by the total tokens in the documents, and then taking log
    normalized_occurrence = math.log(tokens.count(element) / len(tokens))
    # further normalize... NOTE: This line is added for arbitrary reason...
    normalized_occurrence = max(normalized_occurrence, 1)
    # update featureVector by computed TFIDF score accordingly
    TFIDF = normalized_occurrence * tokenIDF[element]
    featureVector[index] = TFIDF

# helper function computing cosine similarity of two feature vectors for modulability 
def cosine_similarity(v1: list, v2: list) -> float:
    # initialize three variables
    sum_xx, sum_xy, sum_yy = 0, 0, 0
    # loop through each list
    for i in range(len(v1)):
        # get current value of v1 and v2
        x, y = v1[i], v2[i]
        # compute the summation
        sum_xx += x * x
        sum_yy += y * y
        sum_xy += x * y
    # try to get square root of the denominator
    try: 
        cosineSimilarity = sum_xy / math.sqrt(sum_xx * sum_yy)
    # if failed, set to 0 directly
    except ZeroDivisionError:
        cosineSimilarity = 0.0
    
    # return the cosine similarity score
    return cosineSimilarity


def corpus_parsing(df: pd.DataFrame) -> dict: 
    # showcase top five rows in df
    # print(df.head(5))

    # initialize variables for: 
    # Python Dictionary to store occurrence of each token
    # Python Dictionary to store IDF for each token
    # Python Dictionary to store the actual corpus <- the returned value
    # counter variable to store the total sentence count
    tokenOccurrence = {}
    tokenIDF = {}
    corpusDict = {}
    row_count = 0

    for row in df['text']:
        row = word_tokenize(row)
        corpusDict[row_count] = [row, [0 for _ in range(len(row))]]
        row_count += 1
    
    # loop through `corpusDict`
    for key, value in corpusDict.items():
        # unpack value
        tokens, featureVector = value
        # remove undesired tokens
        tokens = tokens_cleaner(tokens)
        # loop through each token in `tokens`
        # note that repeated words are intentionally neglected here because:
        # adding them into the model decreases MAP score
        for token in tokens:
            # increment occurrence count accordingly
            tokenOccurrence[token] = tokenOccurrence.get(token, 0) + 1

    # compute the IDF (Inverse Document Frequency) for each token registered in `tokenOccurrence` 
    IDF_updater(row_count, tokenOccurrence, tokenIDF)

    # now compute the TFIDF for each token within its corpus boundary
    for key, value in corpusDict.items():
        # unpack value
        tokens, featureVector = value
        # remove undesired tokens
        keptTokens = tokens_cleaner(tokens)
        # loop through each token in `tokens`
        for index, element in enumerate(tokens):
            # if token is in the kept token list
            if element in keptTokens:
                # compute TFIDF score for it and store at corresponding index
                TFIDF_updater(element, tokens, featureVector, index, tokenIDF)

    return corpusDict

def comment_parsing(comments: list) -> dict: 
    # initialize variables for: 
    # Python Dictionary to store occurrence of each token
    # Python Dictionary to store IDF for each token
    # Python Dictionary to store the actual comment <- the returned value
    # counter variable to store the total sentence count
    tokenOccurrence = {}
    tokenIDF = {}
    commentDict = {}
    row_count = 0

    for row in comments:
        row = word_tokenize(row)
        commentDict[row_count] = [row, [0 for _ in range(len(row))]]
        row_count += 1
    
    # loop through `commentDict`
    for key, value in commentDict.items():
        # unpack value
        tokens, featureVector = value
        # remove undesired tokens
        tokens = tokens_cleaner(tokens)
        # loop through each token in `tokens`
        # note that repeated words are intentionally neglected here because:
        # adding them into the model decreases MAP score
        for token in tokens:
            # increment occurrence count accordingly
            tokenOccurrence[token] = tokenOccurrence.get(token, 0) + 1

    # compute the IDF (Inverse Document Frequency) for each token registered in `tokenOccurrence` 
    IDF_updater(row_count, tokenOccurrence, tokenIDF)

    # now compute the TFIDF for each token within its comment boundary
    for key, value in commentDict.items():
        # unpack value
        tokens, featureVector = value
        # remove undesired tokens
        keptTokens = tokens_cleaner(tokens)
        # loop through each token in `tokens`
        for index, element in enumerate(tokens):
            # if token is in the kept token list
            if element in keptTokens:
                # compute TFIDF score for it and store at corresponding index
                TFIDF_updater(element, tokens, featureVector, index, tokenIDF)

    return commentDict

def cosine_similarity_processing(commentDict: dict, corpusDict: dict) -> dict:
    # initialize output dict
    output = {}

    # loop through the `commentDict`
    for commentKey, commentValue in commentDict.items():
        # initialize comment_output list
        comment_output = []
        # unpack value
        commentTokens, commentFeatureVector = commentValue
        # loop through the `corpusDict`
        for corpusKey, corpusValue in corpusDict.items():
            # unpack value
            corpusTokens, corpusFeatureVector = corpusValue

            # define a new feature vector with same length as the `commentFeatureVector`
            newFeatureVector = [0 for _ in commentFeatureVector]
            # enumerate through `commentTokens`
            for ind, val in enumerate(commentTokens):
                # if the current val (which is a token in the comment) is in `corpusTokens` (corpus's tokens)
                if val in corpusTokens:
                    # add the TFIDF score of that token to the new feature vector
                    newFeatureVector[ind] = corpusFeatureVector[corpusTokens.index(val)]

            # get the cosine similarity between the comment feature vector and the new feature vector
            cosineSimilarity = cosine_similarity(commentFeatureVector, newFeatureVector)
            # append two keys and cosineSimilarity to comment_output list
            if cosineSimilarity > 0:
                # for some reasons there are undesired `0` char on the left of keys under 100, 
                # e.g. 1 would become 001, 10 would become 010... so we left strip the zeros
                comment_output.append((corpusKey, cosineSimilarity))

        # sort the comment_output list after one commentKey is done
        comment_output.sort(key = lambda x: x[1], reverse = True)

        # only extract the top 50 most related reviews
        most_related_50 = []
        for i in range(50):
            most_related_50.append(comment_output[i])
        # add to the output dict
        output[commentKey] = most_related_50
    
    return output

def analyze_vector_similarity(df: pd.DataFrame, comments: list) -> list: 
    cp_s = time()
    corpusDict = corpus_parsing(df)
    cp_e = time()
    print("Time cost for corpus parsing: {0} sec".format(round((cp_e - cp_s), 3)))
    
    cm_s = time()
    commentDict = comment_parsing(comments)
    cm_e = time()
    print("Time cost for comments parsing: {0} sec".format(round((cm_e - cm_s), 3)))

    cs_s = time()
    output = cosine_similarity_processing(commentDict, corpusDict)
    cs_e = time()
    print("Time cost for cosine similarity processing: {0} sec".format(round((cs_e - cs_s), 3)))

    a = time()
    system_output = []
    for key, value in output.items():

        pos = 0
        neg = 0
        for i in value:
            _sentiment = int(df.iloc[i[0]].label)
            if _sentiment == 0: 
                neg += 1
            else: 
                pos += 1

        count_sum = pos + neg
        try: 
            # (positive, negative)
            weight = (round(pos / count_sum, 3), round(neg / count_sum, 3))
        except ZeroDivisionError: 
            weight = (0, 0)

        # print("This comment has weighed sentiment as: \n\tpositive: {0}, negative: {1}"
        #         .format(weight[0], weight[1]))
        system_output.append(weight)
    b = time()
    
    print("Time cost for generating output: {0}.".format(b - a))
    return system_output
    

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
        "His lectures are painful; he reads dense powerpoints with lots of jargon, and it's unclear what's important and what's filler. The assignments are difficult, but more because there's little to no guidance given. The tests are much easier, so this is a relatively easy class but I'd look elsewhere.", 
        "Firstly, Madsen is very knowledgable but he tries to prove this. We were given an Intermediate Micro textbook, however we don't follow it and go far beyond the class. Most lectures do not cover the economics topic but dive into multivariable calc with lots of confusing notation. Feel like I can do math but don't actually understand the economics.", 
        "Short and Simple: Do not take if you care about your GPA. Madsen is not a good professor, and his tests are beyond what is learned in class or homework's.", 
        "This guy does not know the first thing about writing exams. He asked students in his Intermediate Microeconomics to build an ALGORITHM on the exam. Who does that? He is a terrible professor and he does not have the slightest clue about the actual caliber of undergraduate students and thinks we're all PhD students back in Stanford with him.", 
        "Professor Madsen is going to ruin your GPA and you are not going to learn. His homework are extremely difficult and his exams are even more exponentially difficult, impossible even. His exams and homework do not resemble the slides he teaches from, which he did not even make. He is the worst professor I've ever taken in my life."
        ]
    
    bigram_dev_set = '../data/IMDB_data/Valid.csv'
    bigram_train_set = '../data/IMDB_data/Train.csv'
    bigram_test_set = '../data/IMDB_data/Test.csv'

    df = pd.read_csv(bigram_dev_set, header=0)

    comments = []
    with open(bigram_test_set, 'r') as instream: 
        next(instream)
        for line in instream: 
            line = line.strip(os.linesep).split(',')
            try: 
                comments.append(line[0])
            except: 
                continue
    
    print(len(comments))

    analyze_vector_similarity(df, comments) 
