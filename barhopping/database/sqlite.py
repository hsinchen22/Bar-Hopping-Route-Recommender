import sqlite3
from barhopping.config import BARS_DB


def init_bars():
    with sqlite3.connect(BARS_DB) as conn:
        conn.execute(
            '''CREATE TABLE IF NOT EXISTS bars (
               id INTEGER PRIMARY KEY,
               name TEXT, url TEXT, city TEXT,
               address TEXT, rating TEXT,
               photo TEXT, summary TEXT,
               embedding TEXT
             );'''
        )
        conn.commit()


def insert_bar(bar: dict):
    keys   = ",".join(bar.keys())
    qmarks = ",".join("?" * len(bar))
    vals   = list(bar.values())
    with sqlite3.connect(BARS_DB) as conn:
        conn.execute(
            f'INSERT INTO bars({keys}) VALUES({qmarks})', vals
        )
        conn.commit()