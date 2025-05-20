import time, re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin
from bs4 import BeautifulSoup
from barhopping.config import NUM_BARS
from barhopping.config import NUM_PIC
from barhopping.config import NUM_REV
from barhopping.logger import logger

# Single browser instance
def _init_browser():
    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=opts)

browser = _init_browser()

def get_bars(city: str, nums: int = NUM_BARS) -> list[dict]:
    url = f"https://www.google.com/maps/search/bars+in+{city}"
    browser.get(url)
    elems = browser.find_elements(By.CLASS_NAME, "hfpxzc")
    while len(elems) < nums:
        prev = len(elems)
        ActionChains(browser).scroll_from_origin(
            ScrollOrigin.from_element(elems[-1]), 0, 1000
        ).perform()
        time.sleep(2)
        elems = browser.find_elements(By.CLASS_NAME, "hfpxzc")
        if len(elems) <= prev:
            break

    html = browser.page_source
    soup = BeautifulSoup(html, "lxml")
    bars = soup.find_all("a", class_="hfpxzc")
    ratings = soup.find_all("span", class_="MW4etd")
    data = []
    for bar, rate in zip(bars, ratings):
        data.append({
            "name": bar["aria-label"],
            "rating": rate.text,
            "url": bar["href"]
        })
    return data


def get_addr_reviews(url: str, min_char: int = NUM_REV) -> tuple[str, list[str]]:
    browser.get(url)
    time.sleep(1)
    addr = browser.find_element(By.CLASS_NAME, "Io6YTe").text
    # open reviews
    btns = browser.find_elements(By.CLASS_NAME, "hh2c6")
    if len(btns) > 1:
        btns[1].click(); time.sleep(2)

    reviews, count = [], 0
    elems = browser.find_elements(By.CLASS_NAME, "MyEned")
    while count < min_char:
        prev = len(elems)
        ActionChains(browser).scroll_from_origin(
            ScrollOrigin.from_element(elems[-1]), 0, 1000
        ).perform()
        time.sleep(2)
        elems = browser.find_elements(By.CLASS_NAME, "MyEned")
        for m in browser.find_elements(By.CLASS_NAME, "w8nwRe"):
            m.click()
        for e in elems[len(reviews):]:
            txt = re.sub(r"\s+", " ", e.text)
            reviews.append(txt)
            count += len(txt)
            if count >= min_char:
                break
        if len(elems) == prev:
            break
    return addr, reviews


def get_photos(url: str, nums: int = NUM_PIC) -> list[str]:
    browser.get(url)
    try:
        btn = browser.find_element(By.CLASS_NAME, "Dx2nRe")
        btn.click(); time.sleep(1)
        for v in browser.find_elements(By.CLASS_NAME, "hh2c6"):
            if v.text == "Vibe":
                v.click(); time.sleep(1)
        imgs = browser.find_elements(By.CLASS_NAME, "Uf0tqf")
        urls = []
        for img in imgs[:nums]:
            style = img.get_attribute("style")
            urls.append(style[style.find("http"): -3])
        return urls
    except Exception as e:
        logger.error(f"Error getting photos: {e}")
        return []