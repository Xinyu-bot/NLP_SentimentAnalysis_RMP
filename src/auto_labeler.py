import pandas as pd

def main() -> None: 
    ### DO NOT TOUCH ###
    # used to shuffled RMP_data and remove 'index' columns

    df = pd.read_csv('../data/rmp_data/raw/rmp_data_neg2.csv', header=0)
    df.drop(columns='index', inplace=True)
    df.insert(loc=0, column='sentiment', value='')
    df = df.sample(frac=1).reset_index(drop=True)
    df.dropna(how='any', inplace=True)
    
    print(df.head())
    df.to_csv('../data/rmp_data/rmp_data_noIndex_neg.csv', index=False)

    return
    
    
def main2() -> None: 
    # modify filename if needed
    INFILE = '../data/rmp_data/temp.csv' # please modify the temp.csv locally as wish, instead of changing this variable
    OUTFILE = '../data/rmp_data/rmp_data_1234567.csv' # please change this variable directly: if file is created, it will be overrode; if not created, it will be created then

    df = pd.read_csv(INFILE, header=0)
    df_out = pd.DataFrame(columns=['sentiment','text','quality','difficulty','prof_name'])
    print("Total lines in INFILE: {0}".format(len(df.index)))

    counter = 1
    # loop through the set
    for row in df.itertuples(index=False):
        row_d = {'sentiment': row[0], 'text': row[1], 'quality': float(row[2]), 'difficulty': float(row[3]), 'prof_name': row[4]}
        if row_d['quality'] >= 3.0: 
            row_d['sentiment'] = 1
        elif row_d['quality'] <= 2.0: 
            row_d['sentiment'] = 0
        else: 
            label = str(input("=" * 42 + '\n' + "Progress: {0} / {1} = {2}%\n".format(counter, len(df.index), round((counter / len(df.index)), 3) * 100) + "{0}\n".format(row_d['text'])))
            if label != '0' and label != '1': 
                flag = True
                while flag: 
                    label = str(input("MUST BE 0 or 1, please re-enter again\n"))
                    if label == '0' or label == '1':
                        flag = False
            row_d['sentiment'] = label

        counter += 1
        df_out = df_out.append(row_d, ignore_index=True)

    # write out the labeled csv file
    df_out.to_csv(OUTFILE, index=False)

    return

if __name__ == '__main__': 
    main2()