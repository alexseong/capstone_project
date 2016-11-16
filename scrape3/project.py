from bs4 import BeautifulSoup
import requests
import numpy as np


def get_project(url_project):
    '''
    Requests and returns project html.
    '''

    content = requests.get(url_project).content
    soup = BeautifulSoup(content, 'html.parser')
    r = requests.get(url_project)
    status = r.status_code


    print '({0})'.format(status), url_project

    return soup, url_project, status


def get_rewards(url_rewards):
    '''
    Requests and returns rewards html.
    '''
    content = requests.get(url_rewards).content
    soup = BeautifulSoup(content, 'html.parser')
    r = requests.get(url_rewards)
    status = r.status_code

    print '({0})'.format(status), url_rewards

    return soup, url_rewards, status


def get_pledges(rewards):
    '''
    Retrieves the pledge amounts for the project.
    Returns a list of pledge amounts.
    '''
    pledges = rewards.find_all(attrs={'class': 'money usd '})
    if len(pledges) == 0:
        pledges = rewards.find_all(attrs={'class': 'money cad '})
    if len(pledges) == 0:
        pledges = rewards.find_all(attrs={'class': 'money aud '})
        
    pledge_list = []
    for pledge in pledges:
        pledge_list.append(pledge.contents[0][1:])
    return pledge_list

def has_project_video(project):
    '''
    Returns either True or False depending on whether the
    project contains a main video.
    '''
    project_video = project.find_all(attrs={'class': 'project-image'})

    for check in project_video:
        if 'true' in check['data-has-video']:
            return True

        else:
            return False

def count_images(project):
    '''
    Counts the number of pictures provided in the project
    page.
    '''
    images = project.find_all('img', attrs={'class': 'fit'})

    return len(images) + 1

def count_emb_videos(project):
    '''
    Counts the number of embedded videos on the project
    page.
    '''
    hosts = ['youtube', 'vimeo']
    videos = project.find_all(attrs={'class': 'oembed'})
    video_count = 0

    for video in videos:
        video_url = video['data-href']

        if any([host in video_url for host in hosts]):
            video_count += 1

        else:
            pass

    return video_count


def get_tag(project):
    '''
    Retrives the categorical tag for the project. This tag
    is likely a subcategory and will need to be grouped
    later on.
    '''
    try:
        categories = project.find_all(attrs={'class': 'grey-dark mr3 nowrap'})

        category = categories[-1].text.strip()

        return category

    except:
        None

def get_pledges_backed(rewards):
    '''
    Returns a list of strings containing the number of backers for each pledge amounts.
    '''
    backers = rewards.find_all(attrs={'class': 'pledge__backer-count'})
    if len(backers) == 0:
        backers = rewards.find_all(attrs={'class': 'block pledge__backer-count'})
    backers_list = []
    for backer in backers:
        backers_list.append(backer.contents[0].split()[0])

    return backers_list



def get_full_desc(project):
    '''
    Retrieves all the text in the project description.
    '''
    full_desc = project.find_all(class_='full-description')

    desc_str = ''

    for paragraph in full_desc:
        desc_str += paragraph.text.encode('utf-8') + '\n'

    return desc_str
