import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
 

def output_csv(prof_list, output_fn):
    # Output result as a csv file with 4 columns: text, quality, difficulty, prof_name
    df = pd.DataFrame()
    for prof in prof_list:
        comments = prof['comments']
        df_prof = pd.DataFrame([[c[2], c[0], c[1], prof['name']] for c in comments], columns=['text', 'quality', 'difficulty', 'prof_name'])
        df = pd.concat([df, df_prof], ignore_index=True)
    df.to_csv(output_fn)


def retrieve_data():
    prof_list = []  
    base_url = 'https://www.ratemyprofessors.com/ShowRatings.jsp?tid='

    # A list of teacher id whose comments we will fetch
    tid = [x for x in list(range(25000, 26000, 1)) if (x % 100 != 0 and (x % 1000 not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]
    # [1302, 2224004, 1576103, 1051004, 2105994, 2291871, 1482580, 2454762, 1032165, 2291493, 1134872, 1889463, 919428, 2190976, 534980]
    # used: 
    #   (0, 2500000, 100), 
    #   (1, 2500000，1000), 
    #   (9, 2500000),
    #   (5, 2500000)
    #   (0, 26000, 1)

    for t in tid:
        try: 
            url = base_url + str(t)
            prof = get_prof(url)
            prof_list.append(prof)
            print(str(t) + ' success')
        
        except: 
            print(t)
            continue

    print("Total number of professors: " + str(len(prof_list)))
    print("Total number of comments: " + str(sum([len(prof['comments']) for prof in prof_list])))
    output_csv(prof_list, '../data/rmp_data7-6.csv')
    return prof_list


# This function takes a professor name as input, 
# and returns the url of the RMP rating page of this professor
def get_url(prof_name):
    name_parts = prof_name.lower().split()
    query = name_parts[0]
    for i in range(1, len(name_parts)):
        query = query + '+' + name_parts[i]

    # Get the results of searching by this professor name
    query_url = f'https://www.ratemyprofessors.com/search/teachers?query={query}'
    query_res = requests.get(query_url)
    query_dom = BeautifulSoup(query_res.text, features="html.parser")
    
    # If no professor found, exit
    if query_dom.find('div', {'class': 'NoResultsFoundArea__StyledNoResultsFound-mju9e6-0 iManHc'}) != None:
        url = None
    
    # Else
    else:
        # Determine how many professors are found
        num_of_results = int(query_dom.select('div b')[0].text)
        # If only one professor found, get the relative url
        if num_of_results == 1: 
            href = query_dom.find('a', {'class': 'TeacherCard__StyledTeacherCard-syjs0d-0 dLJIlx'})['href']
        # If more than one professor found, prompt the user to select the professor he wants
        else:
            prof_selector = query_dom.find_all('a', {'class': 'TeacherCard__StyledTeacherCard-syjs0d-0 dLJIlx'})
            prof_list = {}
            num = 0
            for prof in prof_selector:
                num = num + 1
                name = prof.find('div', {'class': 'CardName__StyledCardName-sc-1gyrgim-0 cJdVEK'}).text
                department = prof.find('div', {'class': 'CardSchool__Department-sc-19lmz2k-0 haUIRO'}).text
                school = prof.find('div', {'class': 'CardSchool__School-sc-19lmz2k-1 iDlVGM'}).text
                href = prof['href']
                prof_list[num] = {'name': name, 'dpt': department, 'school': school, 'href': href}
            print(f'We found {num_of_results} professors with the name {prof_name}. \nPlease select the one you are referring to:')
            for prof_num in prof_list:
                print(str(prof_num) + ' ' + prof_list[prof_num]['name'] + ', professor of ' + prof_list[prof_num]['dpt'] + ' at ' + prof_list[prof_num]['school'])
            n = input()
            while (not n.isnumeric()) or (n.isnumeric() and (int(n) <= 0 or int(n) > num_of_results)):
                n = input('Please enter a number between 1 and ' + str(num_of_results) + '\n')
            href = prof_list[int(n)]['href']  
        base_url = 'https://www.ratemyprofessors.com'
        url = base_url + href
    return url


# This function takes the url of an RMP rating page, 
# and returns the ratings of the prof as a dict: 
    # {name: Adam Meyers, overall_score: 3.3, would_take_again: 0.67, difficulty: 3, comments: [a_list_of_comments]}
    # "comments" stores all the comments for this professor. Each comment is stored as a list in the form of [quality, difficulty, verbal_comment].
def get_prof(url): 
    try: 
        res = requests.get(url)
        dom = BeautifulSoup(res.text, features="html.parser")

        # Professor name
        name = dom.select("div span")[1].text.strip() + ' ' + dom.select("div span")[2].text.strip()
        
        # Check if this professor has any rating 
        check_finder = dom.find('div', {'class': 'RatingValue__NumRatings-qw8sqy-0 jMkisx'})
        if check_finder != None and check_finder.text[0:10] == 'No ratings':
            prof = {'name': name, 'overall_score': None, 'would_take_again': None, 'difficulty': None, 'comments': None}
            return prof
        
        # Overall quantitative scores of this professor
        overall_score = float(dom.find('div', {'class': 'RatingValue__Numerator-qw8sqy-2 liyUjw'}).text)
        temp_finder = dom.find_all('div', {'class': 'FeedbackItem__FeedbackNumber-uof32n-1 kkESWs'})
        if len(temp_finder) == 1:
            would_take_again = 0
            difficulty = float(temp_finder[0].text)
        else:
            would_take_again = float(temp_finder[0].text.split('%')[0])/100
            difficulty = float(temp_finder[1].text)

        # Get all the comments of this professor
        comments_selector = dom.find('ul', {'class': 'RatingsList__RatingsUL-hn9one-0 cbdtns'}).select('li')
        quality_class = ['CardNumRating__CardNumRatingNumber-sc-17t4b9u-2 kMhQxZ', 'CardNumRating__CardNumRatingNumber-sc-17t4b9u-2 bUneqk', 'CardNumRating__CardNumRatingNumber-sc-17t4b9u-2 fJKuZx']
        difficulty_class = ['CardNumRating__CardNumRatingNumber-sc-17t4b9u-2 cDKJcc']

        comments = []  # A list of all the verbal comments of this professor
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
        return prof
    
    except: 
        print('Something went wrong. Unable to get to get ratings at url: ' + url)


# This function takes url or professor name as input, and returns a list 
# where list[0] is a list of verbal comments, list[1] is the overall score, list[2] is level of difficulty
def get_comments(user_in, mode):
    # mode = 0 if user_in is url
    # mode = 1 if user_in is professor name

    if mode == 0:
        url = user_in   
    else:
        url = get_url(user_in)
        if url == None:
            print('No professor found with name ' + user_in + '.')
            return
    
    # Check the integrity of url
    assert url[0:53] == 'https://www.ratemyprofessors.com/ShowRatings.jsp?tid=', 'Something wrong with the url: ' + url

    prof = get_prof(url)
    comments = None
    if prof['comments'] != None:
        comments = [x[2] for x in prof['comments']]
    return [comments, prof['overall_score'], prof['difficulty'], prof['name']]
