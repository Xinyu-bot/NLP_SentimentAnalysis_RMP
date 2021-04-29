import requests
from bs4 import BeautifulSoup
import pandas as pd

def output_csv(prof_list, output_fn):
    # Output result as a csv file with 4 columns: text, quality, difficulty, prof_name
    df = pd.DataFrame()
    for prof in prof_list:
        comments = prof['comments']
        df_prof = pd.DataFrame([[c[2], c[0], c[1], prof['name']] for c in comments], columns=['text', 'quality', 'difficulty', 'prof_name'])
        df = pd.concat([df, df_prof], ignore_index=True)
    df.to_csv(output_fn)

def main():
    prof_list = []  
    # This program returns prof_list, which is a list of professors with each professor stored as a dictionary.
    # Each professor is stored as a dictionary in the following form:
    # {name: Adam Meyers, overall_score: 3.3, would_take_again: 0.67, difficulty: 3, comments: [a_list_of_comments]}
    # "comments" stores all the comments for this professor. Each comment is stored as a list in the form of [quality, difficulty, verbal_comment].

    base_url = 'https://www.ratemyprofessors.com/ShowRatings.jsp?tid='
    # A list of teacher id whose comments we will fetch

    tid = [x for x in list(range(17000, 18000, 1)) if (x % 100 != 0 and (x % 1000 not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]
    # [1302, 2224004, 1576103, 1051004, 2105994, 2291871, 1482580, 2454762, 1032165, 2291493, 1134872, 1889463, 919428, 2190976, 534980]
    # used: 
    #   (0, 2500000, 100), 
    #   (1, 2500000，1000), 
    #   (9, 2500000),
    #   (5, 2500000)
    #   (0, 16000, 1)

    for t in tid:
        try: 
            url = base_url + str(t)
            res = requests.get(url)
            dom = BeautifulSoup(res.text)

            # Professor name
            name = dom.select("div span")[1].text.strip() + ' ' + dom.select("div span")[2].text.strip()

            # Overall quantitative scores of this professor
            overall_score = float(dom.find('div', {'class': 'RatingValue__Numerator-qw8sqy-2 liyUjw'}).text)
            would_take_again = float(dom.find_all('div', {'class': 'FeedbackItem__FeedbackNumber-uof32n-1 kkESWs'})[0].text.split('%')[0])/100
            difficulty = float(dom.find_all('div', {'class': 'FeedbackItem__FeedbackNumber-uof32n-1 kkESWs'})[1].text)
            
            # Get all the comments of this professor
            comments_selector = dom.find('ul', {'class': 'RatingsList__RatingsUL-hn9one-0 cbdtns'}).select('li')
            quality_class = ['CardNumRating__CardNumRatingNumber-sc-17t4b9u-2 kMhQxZ', 'CardNumRating__CardNumRatingNumber-sc-17t4b9u-2 bUneqk', 'CardNumRating__CardNumRatingNumber-sc-17t4b9u-2 fJKuZx']
            difficulty_class = ['CardNumRating__CardNumRatingNumber-sc-17t4b9u-2 cDKJcc']

            comments = []  # each comment is stored as: [quality, difficulty, verbal_comment]
            for comment in comments_selector:
                if comment.find('div', {'class': difficulty_class[0]}) == None:
                    continue
                for i in range(len(quality_class)):
                    selector = comment.find('div', {'class': quality_class[i]})
                    if selector != None:
                        q = float(selector.text)
                        break
                    else:
                        i += 1
                d = float(comment.find('div', {'class': 'CardNumRating__CardNumRatingNumber-sc-17t4b9u-2 cDKJcc'}).text)
                vc = comment.find('div', {'class': 'Comments__StyledComments-dzzyvm-0 gRjWel'}).text

                # Only get negative comments
                #if q <= 2:
                #    comments.append([q, d, vc])

                # Get all comments
                comments.append([q, d, vc])

            # Each professor is stored as a dictionary in the following form:
            # {name: Adam Meyers, overall_score: 3.3, would_take_again: 0.67, difficulty: 3, comments: [a_list_of_comments]}
            # "comments" stores all the comments for this professor. Each comment is stored as a list in the form of [quality, difficulty, verbal_comment].
            prof = {'name': name, 'overall_score': overall_score, 'would_take_again': would_take_again, 'difficulty': difficulty, 'comments': comments}
            prof_list.append(prof)
            print(str(t) + ' success')
        
        except: 
            print(t)
            continue

        

    print("Total number of professors: " + str(len(prof_list)))
    print("Total number of comments: " + str(sum([len(prof['comments']) for prof in prof_list])))

    output_csv(prof_list, '../data/rmp_data6-8.csv')
    return prof_list

if __name__ == '__main__':
    main()