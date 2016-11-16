from bs4 import BeautifulSoup
import urllib2
import requests
import numpy as np

user_agent = {'User-Agent': 'Mozilla/5.0'}

def get_project(url_project):
    '''
    Requests and returns project html.
    '''
    # url_template = 'https://www.kickstarter.com/projects/{0}/{1}/description'
    #
    # url_project = url_template.format(founder_id, project_id)
    # req = urllib2.Request(url_project, None, headers=user_agent)
    # r = urllib2.urlopen(req)
    # status =  r.getcode()
    # soup = BeautifulSoup(r, 'html.parser')

    content = requests.get(url_project).content
    soup = BeautifulSoup(content, 'html.parser')
    r = requests.get(url_project)
    status = r.status_code


    print '({0})'.format(status), url_project

    return soup, url_project, status


def get_rewards(url_rewards):
    content = requests.get(url_rewards).content
    soup = BeautifulSoup(content, 'html.parser')
    r = requests.get(url_rewards)
    status = r.status_code

    print '({0})'.format(status), url_rewards

    return soup, url_rewards, status


def get_name(project):
    '''
    Retrieves the project name.
    '''
    name = project.find_all(attrs={'class': 'normal mb1'})
    return name[0].text

def get_pledges(rewards):
    '''
    Retrieves the pledge amounts for the project. Returns a
    list of strings containing various pledge amounts.
    '''
    pledges = rewards.find_all(attrs={'class': 'money usd '})
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

# 
# def get_location(project):
#     '''
#     Retrives the city and state where the project is
#     located.
#     '''
#
#     try:
#         locations = project.find_all(attrs={'class': 'grey-dark mr3 nowrap'})
#
#         state = locations[0].text.strip().split(',')[1]
#         city = locations[0].text.strip().split(',')[0]
#
#         return city, state
#     except:
#         return None, None


# def get_blurb(project):
#     '''
#     Retrieves the short project blurb at the top of the
#     project page.
#     '''
#     blurb = project.find_all('p', attrs={'class': 'h3 mb3'})
#
#     try:
#         return blurb[0].text
#
#     except IndexError:
#         return None

def get_full_desc(project):
    '''
    Retrieves all the text in the project description.
    '''
    full_desc = project.find_all(class_='full-description')

    desc_str = ''

    for paragraph in full_desc:
        desc_str += paragraph.text.encode('utf-8') + '\n'

    return desc_str

# def get_goal(project):
#     '''
#     Retrieves the stated project goal.
#     '''
#     goal = project.find_all(class_='money usd no-code')
#
#     try:
#         if len(goal) == 1:
#             return int(goal[0].text[1:].replace(',', ''))
#
#         else:
#             return int(goal[1].text[1:].replace(',', ''))
#     except:
#         return None


# def get_pledged(project):
#     '''
#     Retrieves the amount raised by the project at the end
#     of the project.
#     '''
#     pledged = project.find_all(class_='money usd no-code')
#     try:
#         if len(pledged) == 2:
#             return float(pledged[0].text[1:].replace(',', ''))
#
#         else:
#             pledged_2 = project.find_all(id='pledged')
#
#             return float(pledged_2[0]['data-pledged'].replace(',', ''))
#     except:
#         return None





if __name__ == '__main__':
    project, url, status = get_project(1486216584, 834242902)

    print get_full_desc(project)
