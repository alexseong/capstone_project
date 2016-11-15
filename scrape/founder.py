from bs4 import BeautifulSoup
import urllib2
import requests

user_agent = {'User-Agent': 'Mozilla/5.0'}

def get_profile(url_founder):
    '''
    Requests and scrapes founder profile
    '''

    # req = urllib2.Request(url_founder, None, headers=user_agent)
    # r = urllib2.urlopen(req)
    content = requests.get(url_founder).content

    soup = BeautifulSoup(content, 'html.parser')
    status = requests.get(url_founder).status_code

    print '({0})'.format(status), url_founder

    return soup, url_founder, status

def get_backed(project):
    '''
    Retrieves the number of projects the founder has
    backed.
    '''
    try:
        backed = project.find_all(class_='count')
        return int(backed[0].text[1:-1])
    except:
        return None

def get_created(project):
    '''
    Retrieves the number of projects the founder has
    created.
    '''
    try:
        created = project.find_all(class_='count')
        return int(created[1].text[1:-1])
    except:
        return None

def get_commented(project):
    '''
    Parses the numer of comments the founder has made.
    '''
    try:
        commented = project.find_all(class_='count')
        return int(commented[1].text[1:-1])
    except:
        return None

###########################################################
# founder_bio and get_profile draw from different pages so you will need to use founder_bio in order to use the founder_friends function

def get_bio(founder_id, project_id):
    '''
    Requests and scrapes founder bio, which is different
    from the profile page.
    '''

    url_template = 'https://www.kickstarter.com/projects/{0}/{1}/creator_bio'

    url_bio = url_template.format(founder_id, project_id)
    # req = urllib2.Request(url_bio, None, headers=user_agent)
    # r = urllib2.urlopen(req)
    # status = r.getcode()

    content = requests.get(url_bio).content
    soup = BeautifulSoup(content, 'html.parser')
    status = requests.get(url_bio).status_code

    print '({0})'.format(status), url_bio

    return soup, url_bio, status



def founder_friends(project):
    '''
    Retrieves the number of Facebook friends the founder
    has if Facebook account is connect to Kickstarter
    account.
    '''

    friends = project.find_all(class_='facebook py1 border-bottom h5')
    try:
        return int(''.join(c for c in friends[0].text if c.isdigit()))
    except:
        return None
