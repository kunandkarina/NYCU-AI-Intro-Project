from bs4 import BeautifulSoup
from scrapy.selector import Selector
import requests
import re
import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from tqdm import tqdm

# 設定Chrome Driver的執行檔路徑
options = Options()
options.chrome_executable_path="C:\\Users\\user\\Desktop\\code\\python\\project\\chromedriver.exe"
# 建立 Driver物件實體，讓程式操作瀏覽器運作
driver = webdriver.Chrome(options=options)
# 要爬蟲的網頁
url = 'https://www.imdb.com/title/tt10954600/reviews?ref_=tt_urv'
driver.get(url)

page = 1
# IMDB中每個頁面只有25則評論，因此我們必須翻10頁來取得200筆以上的資訊
while page < 10:
    try:
        # 用css_selector找尋'load-more-trigger'的位置
        css_selector = 'load-more-trigger'
        # 使用Driver物件自動點擊
        driver.find_element(By.ID, css_selector).click()
        time.sleep(3)
        page += 1
    except:
        pass
# 尋找class = review-container的標籤
review = driver.find_elements(By.CLASS_NAME, 'review-container')
# 儲存星星數與評價的list
rating = []
lis = []
cnt = 0
# 設定最多找200筆資訊
for n in range(0,250):
    try:
        if cnt >=200:
            break
        # 用戶評論必須同時具備rating和title的資料，否則略過並尋找下一筆
        frating = review[n].find_element(By.CLASS_NAME, 'rating-other-user-rating').text
        flist = review[n].find_element(By.CLASS_NAME, 'title').text

        rating.append(frating)
        lis.append(flist)
        cnt += 1
    except:
        continue
# 將rating的資料從string轉成int
for i in range(len(rating)):
    rating[i] = rating[i].replace('/10', "")
    rating[i] = int(rating[i])

print(rating)
print(lis)
# 這邊是將每部電影的200則評論存入csv檔，用來確認有取得資訊。
data = {'Rate' : rating, 'Review' : lis}
result = pd.DataFrame(data=data)
result.to_csv('result.csv',encoding="utf_8_sig")