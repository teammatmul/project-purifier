from bs4 import BeautifulSoup 
from urllib.request import urlopen
from selenium import webdriver
from pandas import Series, DataFrame
import numpy as np
import pandas as pd
import re
import time

"""
namuwiki 랜덤 페이지 클릭으로 크롤링하는 코드입니다.
namu1 = namuwiki_crwal(hom_many)
hom_many 만큼 클릭해서 크롤링합니다.
"""

def make_beautiful_text(soup):
    result = []
    temp = soup.find_all('div', 'wiki-heading-content')
    for i in range(len(temp)):
        temp1 = temp[i].get_text()
        if 'img class' in temp1: pass
        else:
            temp1 = re.sub("\[\d{1,3}\]",'',temp1)
            temp1 = re.sub("\[A\]",'',temp1)
            temp1 = re.split('\.\ ?', temp1)
            for j in range(len(temp1)):
                if temp1[j] == '': pass
                elif len(temp1[j]) < 20 : pass
                else: result.append(temp1[j])
    return result

def namuwiki_crwal(how_many):
    crwal_text = []
    driver_path = "path\\chromedriver.exe"
    driver = webdriver.Chrome(executable_path=driver_path)
    url = 'https://namu.wiki'
    driver.get(url)
    while how_many !=0:
        try:
            #랜덤 버튼 클릭
            driver.find_element_by_css_selector('body > div.navbar-wrapper > nav > form > div > span.input-group-btn.left-search-btns > a').click()
            time.sleep(2)
            #페이지 html 긁어오기
            html = driver.page_source
            soup = BeautifulSoup(html, "lxml")
            #html에서 text 추출 함수 사용
            result = make_beautiful_text(soup)
            #crwal_text에 넣어주기
            crwal_text.extend(result)
            #count -1
            how_many -= 1
        except:
            print('한번 넘어감')
            driver.refresh()
            time.sleep(5)
            how_many -= 1
            pass
    namu_crwal = pd.DataFrame({'Comments': crwal_text})
    print('-----한회차 끝------')
    return namu_crwal

# 나무위키 크롤링 100자 이내로 한문장을 잘라내는 코드입니다.
def sliceString(x):
    if len(str(x))>100: x = str(x)[:100]
    return x