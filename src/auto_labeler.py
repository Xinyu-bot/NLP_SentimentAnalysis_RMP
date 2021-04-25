import os
import pandas as pd

def main() -> None: 
    '''
    ### DO NOT TOUCH ###

    df = pd.read_csv('../data/rmp_data/rmp_data.csv', header=0)
    df.drop(columns='index', inplace=True)
    df.insert(loc=0, column='sentiment', value='')
    df = df.sample(frac=1).reset_index(drop=True)
    df.dropna(how='any', inplace=True)
    
    print(df.head())
    df.to_csv('../data/rmp_data/rmp_data_noIndex.csv', index=False)
    '''

    # modify filename if needed
    INFILE = '../data/rmp_data/rmp_data_noIndex.csv'
    OUTFILE = '../data/rmp_data/rmp_data_altesting.csv'
    LINES_TO_LABEL = 500

    df = pd.read_csv(INFILE, header=0)
    df_out = pd.DataFrame(columns=['sentiment','text','quality','difficulty','prof_name'])

    counter = 0
    # loop through the set
    for row in df.itertuples(index=False):
        row_d = {'sentiment': row[0], 'text': row[1], 'quality': float(row[2]), 'difficulty': float(row[3]), 'prof_name': row[4]}
        if row_d['quality'] >= 3.0: 
            row_d['sentiment'] = 1
        elif row_d['quality'] <= 2.0: 
            row_d['sentiment'] = 0
        else: 
            label = str(input("{0}\n".format(row_d['text'])))
            if label != '0' and label != '1': 
                flag = True
                while flag: 
                    label = str(input("MUST BE 0 or 1, please re-enter again\n"))
                    if label == '0' or label == '1':
                        flag = False
            row_d['sentiment'] = label

        df_out = df_out.append(row_d, ignore_index=True)
        counter += 1
        if counter == LINES_TO_LABEL: 
            break

    df_out.to_csv(OUTFILE, index=False)
    
    return

if __name__ == '__main__': 
    main()