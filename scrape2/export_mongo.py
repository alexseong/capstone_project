import pandas as pd
from pymongo import MongoClient

client = MongoClient()
db = client['kickstarter']
collection = db['data']

def read_mongo(collection, query={}):
    '''
    Converts MongoDB collection into pandas DataFrame.
    '''
    cursor = collection.find(query)

    df = pd.DataFrame(list(cursor))

    try:
        del df['_id']
    except:
        pass

    return df

if __name__ == '__main__':
    read_mongo(collection).to_csv('kickstarter.csv', encoding = 'utf-8')
