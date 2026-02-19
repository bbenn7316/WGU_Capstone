import pandas as pd
import numpy as np
import os
import pathlib



#Dictionary Provided By: Adrian 124 [1]
column_name_dict = {
"name": "Name",
"release_date": "Release date",
"required_age": "Required age",
"price": "Price",
"dlc_count": "DLC count",
"detailed_description": "Detailed description",
"about_the_game": "About the game",
"short_description": "Short description",
"reviews": "Reviews",
"header_image": "Header image",
"website": "Website",
"support_url": "Support url",
"support_email": "Support email",
"windows": "Windows",
"mac": "Mac",
"linux": "Linux",
"metacritic_score": "Metacritic score",
"metacritic_url": "Metacritic url",
"achievements": "Achievements",
"recommendations": "Recommendations",
"notes": "Notes",
"supported_languages": "Supported languages",
"full_audio_languages": "Full audio languages",
"packages": "Packages",
"developers": "Developers",
"publishers": "Publishers",
"categories": "Categories",
"genres": "Genres",
"screenshots": "Screenshots",
"movies": "Movies",
"user_score": "User score",
"score_rank": "Score rank",
"positive": "Positive",
"negative": "Negative",
"estimated_owners": "Estimated owners",
"average_playtime_forever": "Average playtime forever",
"average_playtime_2weeks": "Average playtime two weeks",
"median_playtime_forever": "Median playtime forever",
"median_playtime_2weeks": "Median playtime two weeks",
"peak_ccu": "Peak CCU",
"tags": "Tags"
}

# Function Provided By: Adrian 124 [1]
def convert_dict_to_string(dict_object):
    # Used for converting the .json data into the format used in the .csv file
    # i.e. dict of "Tag: tag_id" into string with comma-separated tags
    if len(dict_object) == 0:
        return np.nan
    key_list = list(dict_object.keys())
    keys_string = ",".join(key_list)
    return keys_string


# Function Provided By: Adrian 124 [1]
def read_convert_json_dataset():
    json_path = pathlib.Path(__file__).parent / 'data' / 'games.json'
    _df = pd.read_json(json_path)
    _df = _df.T
    _df['AppID'] = _df.index
    _df.rename(columns=column_name_dict, inplace=True)

    # Convert the dict/array columns
    for col in _df.columns.values:
        if isinstance(_df.loc[546560][col], dict):
            _df[col] = _df[col].apply(lambda entries: convert_dict_to_string(entries))

    _df = _df.reset_index().set_index("AppID")
    return _df

def cleanup_df(df):
    df['Genres'] = df['Genres'].apply(', '.join)
    for x in df.index:
        if df.at[x, 'Positive'] > 1000000:
            df.at[x, 'Positive'] = 1000000
        if df.at[x, 'Peak CCU'] > 100000:
            df.at[x, 'Peak CCU'] = 100000
        if df.at[x, 'Average playtime forever'] > 100000:
            df.at[x, 'Average playtime forever'] = 100000
        if df.at[x, 'Recommendations'] > 1000000:
            df.at[x, 'Recommendations'] = 1000000
    df2 = df.dropna()
    del df
    return df2

def select_genre(genre, df):
    if genre == "":
        return df
    else:
        genre_df = df[df['Genres'].str.contains(genre)]

    return genre_df