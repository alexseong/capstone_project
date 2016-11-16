import pandas as pd
import numpy as np

'''
The functions below clean the Webrobot
datasets which can be found at: http://webrobots.io/kickstarter-datasets/

'''

column_names = [u'_id', u'created_at',
                u'data', u'id',
                u'run_id', 'project_id',
                'creator_id', 'creator_name',
                'goal', 'name',
                'slug', 'blurb',
                'status', 'pledged',
                'cat_name', 'cat_slug',
                'currency', 'short_name',
                'state', 'country',
                'spotlight', 'staff_pick',
                'launched_at', 'usd_pledged',
                'backers_count', 'currency_symbol',
                'static_usd_rate', 'state_changed_at',
                'disable_communication', 'duration_to_launch',
                'proj_live_duration']


# read the entire file into a python array
def read_file(filepath):
    with open(filepath, 'rb') as f:
        data = f.readlines()

    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)

    # each element of 'data' is an individual JSON object.
    # convert it into an *array* of JSON objects, which is one large JSON object
    # translation: add square brackets to the beginning and end,
    # and have all the individual business JSON objects separated by a comma

    data_json_str = "[" + ','.join(data) + "]"

    # load it into pandas
    df = pd.read_json(data_json_str)
    return df

# remove columns
def remove_features(df):
    df = df.drop('table_id', axis=1)
    df = df.drop('robot_id', axis = 1)
    return df

# extract features within nested json object ['data']

def extract_data_features(df):
    df['status'] = df['data'].map(lambda x: x['state'])

    df['project_id'] = df['data'].map(lambda x: x['id'])
    df['creator_id'] = df['data'].map(lambda x: x['creator']['id'])
    df['creator_name'] = df['data'].map(lambda x: x['creator']['name'])

    df['goal'] = df['data'].map(lambda x: x['goal'])
    df['name'] = df['data'].map(lambda x: x['name'])
    df['slug'] = df['data'].map(lambda x: x['slug'])
    df['blurb'] = df['data'].map(lambda x: x['blurb'])

    df['pledged'] = df['data'].map(lambda x: x['pledged'])
    df['cat_name'] = df['data'].map(lambda x: x['category']['name'])
    df['cat_slug'] = df['data'].map(lambda x: x['category']['slug'])
    df['currency'] = df['data'].map(lambda x: x['currency'])

    df['loc_type'] = df['data'].map(lambda x: x['location']['type'])
    df['short_name'] = df['data'].map(lambda x: x['location']['short_name'])
    df['state'] = df['data'].map(lambda x: x['location']['state'])
    df['country'] = df['data'].map(lambda x: x['location']['country'])
    df['spotlight'] = df['data'].map(lambda x: x['spotlight'])
    df['staff_pick'] = df['data'].map(lambda x: x['staff_pick'])
    df['created_at'] = df['data'].map(lambda x: x['created_at'])
    df['launched_at'] = df['data'].map(lambda x: x['launched_at'])
    df['usd_pledged'] = df['data'].map(lambda x: x['usd_pledged'])

    df['backers_count'] = df['data'].map(lambda x: x['backers_count'])
    df['currency_symbol'] = df['data'].map(lambda x: x['currency_symbol'])
    df['static_usd_rate'] = df['data'].map(lambda x: x['static_usd_rate'])
    df['state_changed_at'] = df['data'].map(lambda x: x['state_changed_at'])
    df['disable_communication'] = df['data'].map(lambda x: x['disable_communication'])

    return df

# convert int64 time to datetime and find interval features

 def get_interval(df):
    df['created_at'] = pd.to_datetime(df.created_at, unit='s')
    df['launched_at'] = pd.to_datetime(df.launched_at,unit='s')
    df['state_changed_at'] = pd.to_datetime(df.state_changed_at,unit='s')

    df['created_at'] = df['created_at'].map(lambda x: x.date())
    df['launched_at'] = df['launched_at'].map(lambda x: x.date())
    df['state_changed_at'] = df['state_changed_at'].map(lambda x: x.date())

    # find interval features
    df['days_to_launch'] = (df['launched_at'] - df['created_at']).map(lambda x:x.days)
    df['proj_live_days'] = (df['state_changed_at'] - df['launched_at']).map(lambda x:x.days)
    return df
#######################################################

def extract_category(df):
    '''
    Extracts project category name and id from the nested
    json object in the 'category' feature of the df.
    '''
    category_names = []
    category_ids = []

    for category in df.category.values:
        try:
            category_names.append(category['name'])
            category_ids.append(category['id'])
        except TypeError:
            category_names.append(None)
            category_ids.append(None)


    df['category_name'] = np.array(category_names)
    df['category_id'] = np.array(category_ids)

    return df

def get_webrobot(df):
    '''
    Returns dataframe with desired features. These features
    will be joined into the final dataset.
    '''
    return df[columns_2]



if __name__ == '__main__':
    # df = read_file('full_data.json')
    # df = remove_features(df)
    # df = extract_data_features(df)
    # df = get_interval(df)
