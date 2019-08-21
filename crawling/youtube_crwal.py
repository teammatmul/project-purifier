from bs4 import BeautifulSoup
from urllib.request import urlopen
from datetime import datetime, timedelta
from collections import Counter

import pandas as pd
import re
import time

# 크롬 브라우저 조작을 위한 모듈
from selenium import webdriver

# 페이지 스크롤링을 위한 모듈
from selenium.webdriver.common.keys import Keys

# url 가져오기(인기 영상)
url = 'https://www.youtube.com/feed/trending'

# 페이지 이동
driver_path = "./crwal/driver/chromedriver.exe"
driver = webdriver.Chrome(executable_path=driver_path)
time.sleep(5)
driver.get(url)
time.sleep(10)

# 페이지 소스 가져오기
data = BeautifulSoup(driver.page_source, "lxml")

# comment_urls 만들기
comment_urls = []

for x in range(len(data.find_all('a', 'yt-simple-endpoint style-scope ytd-video-renderer'))):
    comment_urls.append('https://www.youtube.com/'+ data.find_all('a', 'yt-simple-endpoint style-scope ytd-video-renderer')[x]['href'])


# 댓글 가져오기
comment_sum = []
title_sum = []
url_num = 0

# 전체 시작시간 체크
start_total = datetime.now()

for url in comment_urls:

    # comment_url 로 이동하기
    driver.get(url)
    time.sleep(5)

    # try 문(오류시 계속 진행하기 위함)
    try:
        # 동영상 일시정지 누르기
        driver.find_element_by_xpath("""//*[@id="movie_player"]/div[1]/video""").click()

        # 페이지 소스 및 제목 가져오기
        comment = BeautifulSoup(driver.page_source, "lxml")
        title = comment.find('yt-formatted-string', 'style-scope ytd-video-primary-info-renderer').get_text()

        control = driver.find_element_by_tag_name("body")
        SCROLL_PAUSE_TIME = 10
        step = 1

        # 시작시작 체크
        start = datetime.now()

        # url_num 1증가
        url_num += 1

        # 댓글 시작 알림
        print("=" * 50)
        print("댓글 수집을 시작합니다. Title<%d> : %s" % (url_num, title))

        # scroll 끝까지 내리기
        control.send_keys(Keys.PAGE_DOWN)
        control.send_keys(Keys.PAGE_DOWN)
        control.send_keys(Keys.PAGE_DOWN)
        time.sleep(3)

        # Get comment_count
        last_comment_count = len(comment.find_all('yt-formatted-string', id='content-text'))

        while True:
            # Scroll down to bottom
            for x in range(4):
                control.send_keys(Keys.PAGE_DOWN)

            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)

            # Calculate new_comment_count and compare with last_comment_count
            comment = BeautifulSoup(driver.page_source, "lxml")
            new_comment_count = len(comment.find_all('yt-formatted-string', id='content-text'))

            # end comment check
            if new_comment_count == last_comment_count:
                break

            print("step = %d, last_comment_count = %d, new_comment_count = %d" % (
            step, last_comment_count, new_comment_count))
            step += 1

            last_comment_count = new_comment_count

        # comment 가져오기
        comment = comment.find_all('yt-formatted-string', id='content-text')

        for y in range(new_comment_count):
            comment_sum.append(
                comment[y].get_text().replace('\ufeff', '').replace('\n', '').replace('\w', '').replace('\r', ''))
            title_sum.append(title)

        end = datetime.now()
        print("=" * 50)
        print("댓글 수집을 완료했습니다.")
        elapsed = end - start
        elapsed_total = end - start_total
        print('이번 계산 시간 : ', end='');
        print(elapsed)
        print('이번 수집 개수 : %d' % new_comment_count)
        print('현재까지 총 계산 시간 : ', end='');
        print(elapsed_total)
        print('현재까지 총 수집 개수: %d개' % (len(comment_sum)))

    # except 문(오류시 다음 ulr로 넘어가서 진행)
    except:
        print('Error가 발생했습니다. 다음 url로 넘어갑니다.')
        continue

print("=" * 50)
print("모든 수집이 끝났습니다. 축하합니다. ^^")

