import pandas as pd
import numpy as np
import random
import time
import project
import founder
from bs4 import BeautifulSoup
from pymongo import MongoClient

# Initialize MongoDB database and collection
client = MongoClient()
db = client['kickstarter']
collection = db['subdata']

def soupify(string):
    '''
    Converts html string to BeautifulSoup object.
    '''
    return BeautifulSoup(string, 'html.parser')

def featurize(project_id, founder_id, project_url, founder_url, rewards_url, project_soup, founder_soup, rewards_soup, collection):
    '''
    Parses project and founder html for features and
    inserts them into a MongoDB database.
    '''

    features = {'founder_id': int(founder_id),
                'project_id': int(project_id),
                'project_url': project_url,
                'founder_url': founder_url,
                'pledges': project.get_pledges(rewards_soup),
                'pledge_backer_count' : project.get_pledges_backed(rewards_soup),
                'main_video': project.has_project_video(project_soup),
                'image_count': project.count_images(project_soup),
                'emb_video_count': project.count_emb_videos(project_soup),
                'founder_backed': founder.get_backed(founder_soup),
                'founder_created': founder.get_created(founder_soup),
                'founder_comments': founder.get_commented(founder_soup),
                'tag': project.get_tag(project_soup),
                'description': project.get_full_desc(project_soup)}



    collection.insert(features)
