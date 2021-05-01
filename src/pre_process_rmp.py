import os
import pandas as pd

def shuffle() -> None: 
    INFILE = '../data/rmp_data/processed/rmp_data_labeled.csv' 
    train_set = '../data/rmp_data/processed/rmp_data_train.csv'
    test_set = '../data/rmp_data/processed/rmp_data_test.csv'

    df = pd.read_csv(INFILE, header=0)
    df.drop_duplicates(inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)
    row_count = df['label'].count()
    print(df['label'].value_counts())
    

    df = df.sort_values('label')
    SIZE = int(input('SIZE = \t'))
    pos = df.head(SIZE)
    neg = df.tail(SIZE)

    total = SIZE * 2
    out = pos.append(neg, ignore_index=True)
    out = out.sample(frac=1).reset_index(drop=True)
    test_size = int(total * 0.2 // 1)
    training_size = total - test_size
    


    df_train = out.head(training_size)
    df_train.to_csv(train_set, index=False)

    df_test = out.tail(test_size)
    df_test.to_csv(test_set, index=False)
    return

def main() -> None:
    # modify filename if needed
    INFILE = '../data/rmp_data/rmp_data_6.csv' 
    OUTFILE = '../data/rmp_data/rmp_data_reversed.csv'

    df = pd.read_csv(INFILE, header=0)
    data = {'text': df['text'], 'label': df['sentiment']}
    df_out = pd.DataFrame(data)
    
    df_out.to_csv(OUTFILE, index=False)

    return

if __name__ == '__main__':
    # main()
    shuffle()