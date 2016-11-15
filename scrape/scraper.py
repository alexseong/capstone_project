import pandas as pd
import numpy as np
import random
import time
import project
import founder
from bs4 import BeautifulSoup
from project import get_project, get_rewards
from founder import get_profile, get_bio
from featurizer1 import featurize
from pymongo import MongoClient
import urllib2


# Set user_agent to Mozilla browser
user_agent = {'User-Agent': 'Mozilla/5.0'}

# Initialize MongoDB database and collection
client = MongoClient()
db = client['ksdb']
collection = db['ksdata']

# Load array of project_id, founder_id
# url_list = np.loadtxt('url_list.csv', delimiter=',')
df = pd.read_csv('id_url_list.csv', dtype=object, header=None)
id_url_list = df.values
#
# def timeit(method):
#     '''
#     Wrapper for timing functions.
#     '''
#     def timed(*args, **kw):
#         ts = time.time()
#         result = method(*args, **kw)
#         te = time.time()
#
#         print '%r %2.2f sec' % \
#               (method.__name__, te-ts)
#
#         return result
#
#     return timed

def subsample(arr, p=1):
    '''
    Returns random subsample of 2D array. Default sample
    size is 100.
    '''
    mask = np.random.choice([True, False], arr.shape[0],
        p = [p, 1 - p])

    sub = arr[mask]

    return sub

def id_generator(arr):
    '''
    Create new generator.
    '''
    return (x for x in arr)

# @timeit
def extract(generator):
    '''
    Scrapes Kickstarter pages and parses features into
    MongoDB database. This function calls the featurize
    function from the featurizer module to insert data
    into the MongoDB database.
    '''
    go = True
    progress = 0
    skipped = 0
    failed = 0

    while go:
        block_size = random.randint(5, 10)
        wait_time = random.randint(2, 4)
        wait = False

        print '\n'
        print 'Block size: {0}'.format(block_size)

        for i in xrange(0, block_size):
            # founder_id, project_id = (int(x) for x in generator.next())
            project_id,founder_id, project_url, founder_url, rewards_url = (x for x in generator.next())

            collection_check = set(db.ksdata.distinct('project_id', {}))
            if project_id in collection_check:
                    print "Already scraped"
                    skipped += 1
                    wait = False


            else:
                try:
                    project_soup, project_url, status1 = get_project(project_url)

                    founder_soup, founder_url, status2 = get_profile(founder_url)

                    rewards_soup, rewards_url, status3 = get_rewards(rewards_url)

                    if (status1 & status2 & status3) == 200:
                        featurize(project_id, founder_id, project_url, founder_url, rewards_url, project_soup, founder_soup, rewards_soup, collection)

                        progress += 1

                        wait = True

                except requests.ConnectionError:
                    failed +=1

        print '\n'

        print 'Scraped: {}'.format(progress)

        print 'Skipped: {}'.format(skipped)

        print 'Failed: {}'.format(failed)

        print 'Next block: {}s'.format(wait_time)

        if wait:
            time.sleep(wait_time)

        else:
            pass
