from bs4 import BeautifulSoup
import requests


def get_profile(url_founder):
    '''
    Requests and scrapes founder's profile
    '''

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


# def founder_friends(project):
#     '''
#     Retrieves the number of Facebook friends the founder
#     has if Facebook account is connected to Kickstarter
#     account.
#     '''
#
#     friends = project.find_all(class_='facebook py1 border-bottom h5')
#     try:
#         return int(''.join(c for c in friends[0].text if c.isdigit()))
#     except:
#         return None
