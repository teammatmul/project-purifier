# 일베 일간 베스트 크롤링 코드
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import re
import pandas as pd

def crawling_comment(title, titleList, commentList):
    # 게시글 페이지 파싱
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
    
    comments = soup.find_all('div', {'class' : re.compile('comment_\d+ xe_content')})
    for n in range(len(comments)):
        titleList.append(title)
        commentList.append(comments[n].get_text().replace('\n',''))
        
driver_path = "driver/chromedriver.exe"
driver = webdriver.Chrome(executable_path=driver_path)

# 일베 일간 베스트 페이지 이동
url_page = 'http://www.ilbe.com/ilbe'
driver.get(url_page)

Titles = []
Comments = []
page = 1
pageNum = 4

# 일베 main 크롤링
while page < 100:
    # 타이틀 페이지 파싱
    html1 = driver.page_source
    titlePage = BeautifulSoup(html1, "lxml")
        
    # 해당 페이지의 모든 제목을 담음
    tempTitles = titlePage.find_all('td', {'class' : re.compile('title bdoc_\d+')})
    
    for n in range(22):
        # 이미 크롤링한 제목이면 패스
        tempTitle = tempTitles[n].find('a').get_text().replace('\n','')
        if tempTitle in Titles: continue
        else:
            while True:
                try:
                    print(n, tempTitle)
                    # 해당 게시글로 이동
                    time.sleep(1)
                    driver.find_element_by_xpath('//*[@id="content"]/div[1]/form/table/tbody/tr[{}]/td[2]/a'.format(n+5)).click()
                    break
                except:
                    break
            
            # 댓글 크롤링 함수 호출
            crawling_comment(tempTitle, Titles, Comments)
            driver.back()
    
    # 다음 페이지 이동
    while True:
        try:
            time.sleep(2)
            if pageNum < 9:
                driver.find_element_by_xpath('//*[@id="content"]/div[1]/div[4]/div[3]/a[{}]'.format(pageNum)).click()
                pageNum += 1
            else:
                driver.find_element_by_xpath('//*[@id="content"]/div[1]/div[4]/div[3]/a[{}]'.format(pageNum)).click()
            page += 1
            print('{}페이지 크롤링 완료, 댓글 {}개 수집'.format(page, len(Comments)))
            break
        except:
            break

