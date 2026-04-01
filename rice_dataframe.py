import pandas as pd

def create_df(features):
    df = pd.DataFrame(features)
    return df

def classify(df, length_threshold=150, aspect_ratio_threshold=2.5):
    # Classify using both length and shape
    def get_category(row):
        if row['Length'] >= length_threshold and (row['Length'] / row['Width']) >= aspect_ratio_threshold:
            return 'Long Rice'
        else:
            return 'Short Rice'

    df['Category'] = df.apply(get_category, axis=1)
    return df

def save_to_csv(df, filename='rice_grains.csv'):
    df.to_csv(filename, index=False)