#!/usr/bin/env python3
"""
REDDIT: ОБГОВОРЕННЯ МІЖНАРОДНОЇ ПІДТРИМКИ УКРАЇНИ
Фільтрація та підготовка ОДНОГО датасету (2019 — до війни, 2022 — повномасштабне вторгнення)

Вхід:  CSV-файли, які створює парсер:
       output/reddit_YYYY-MM_submissions.csv та output/reddit_YYYY-MM_comments.csv
Вихід: data/reddit_ukraine_support.csv  — єдиний датасет
       data/preview_by_month.csv         — ~40 випадкових рядків за кожен місяць, щоб
                                           переглянути дані очима (відкривається в PyCharm цілком)
       data/validation_sample.csv        — вибірка для ручної перевірки якості фільтра

Логіка (три рівні замість балів):
  1. Чи текст про Україну/Росію?   Джерело тематичності записується в topic_source:
       text_ukraine  — текст прямо згадує Україну (міста, лідерів, Крим, Донбас…)
       text_russia   — текст згадує Росію разом із маркером агресії (invasion, annexation,
                       occupation, sanctions, mobilization…) і не про сторонні теми
                       (Сирія, Мюллер, Трамп і Росія, Скрипалі тощо)
       thread        — коментар у треді, заголовок якого тематичний за тим самим правилом
       subreddit     — тематичний сабреддіт (див. DEDICATED_SUBREDDITS)
     Для thread і subreddit сам коментар має містити власний сигнал: Україну, Росію, маркер
     агресії або тип підтримки; назва країни чи організації рахується лише разом з українським
     посадовцем або описом підтримки («send them more weapons»). Інакше («What do you mean?»,
     «lol», «that tank got wrecked», «indoor plumbing in France?») коментар відкидається
     (див. CONTEXT_NEEDS_TOPIC_SIGNAL).
  2. Які країни / організації згадано і як (терміни, речення-контекст).
     countries_focus — ті з них, що стоять в одному реченні з Україною, типом підтримки
     або з Росією й маркером агресії (суворіший варіант для аналізу).
  3. Про що саме: тип підтримки — і для тексту загалом, і для кожної країни окремо.

Жорстко видаляються: боти й повідомлення модераторів, записи поза періодом, тексти,
коротші за MIN_WORDS слів після очищення, дублікати (той самий id або той самий текст
від того самого автора), коментарі з тредів/сабреддітів без власного сигналу.
Усе інше зберігається з прапорцями (us_domestic, is_lifestyle тощо).

Колонки датасету:
    id, type, subreddit                    — запис (пост / коментар) і сабреддіт
    date                                   — дата (UTC)
    author_country, score                  — країна автора (флер / «As an American…»), рейтинг
    thread_id, thread_n                    — тред і скільки рядків із нього в датасеті
                                             (заголовок треду = text поста з id == thread_id)
    text                                   — очищений текст
    topic_source                           — чому запис потрапив у датасет (див. вище)
    us_domestic, is_lifestyle              — внутрішня політика США / побут: виключати з аналізу
    countries, organizations,              — згадані країни, NATO / EU / UN (через «|»);
      countries_focus                        ті з них, що в одному реченні з Україною чи підтримкою
    aid_military … membership (8 колонок)  — тип підтримки в тексті (1/0)
    country_support                        — тип підтримки окремо для кожної країни (JSON)
    subreddit_day_total,                   — скільки ВСЬОГО текстів було в сабреддіті за день /
      subreddit_month_total                  місяць (знаменник для порівняння 2019 і 2022)
    reaction_ukraine, reaction_russia,     — реакція на Україну / Росію / кожного адресата
      reaction                               (1 позитив, 0 нейтрал, −1 негатив; reaction — JSON
                                             {"Poland":1,"Russia":-1}); дописує add_reactions.py

Колонки, яких у файлі немає (вони однозначно рахуються з інших), — load_dataset() додає їх сам:
    week                — понеділок тижня (з date)
    month               — «2022-03» (з date)
    period              — 2019 / 2022_pre / 2022_war (з date)
    days_since_invasion — днів від 24.02.2022, для 2019 порожньо (з date)
    mentions_ukraine    — текст прямо згадує Україну (= topic_source == "text_ukraine")
    Без load_dataset:  df["month"] = df["date"].str[:7]

Посилання на запис (колонки permalink у файлі немає, щоб він був меншим):
    коментар:  https://www.reddit.com/r/{subreddit}/comments/{thread_id}/_/{id}/
    пост:      https://www.reddit.com/r/{subreddit}/comments/{id}/
Прапорці (us_domestic, aid_military, …) записані як 1/0.

Запуск:  python filter_reddit.py
Потрібно: pandas, numpy, tqdm   (pip install pandas numpy tqdm; tqdm — лише для смужки прогресу)

Як працювати з результатом:
    from filter_reddit import load_dataset      # читає CSV з правильними типами
    df = load_dataset()                         # списки → списки, дати → дати, + week/month/period
    df = load_dataset(usecols=["id", "period", "countries", "aid_military"])   # менше пам'яті
                                                # (period порахується з date автоматично)

    # Частка тематичних текстів від усієї активності сабреддіту (для порівняння 2019 і 2022)
    m = df.groupby(["subreddit", "month", "type"]).agg(
        n=("id", "size"), total=("subreddit_month_total", "first"))
    m["share"] = m["n"] / m["total"]

    # Один рядок на пару «текст × країна»
    long = df[df["countries"].map(len) > 0].explode("countries")

    # Суворіший варіант: лише країни в одному реченні з Україною / підтримкою / агресією;
    # без внутрішньої політики США; мегатреди отримують меншу вагу
    core = df[~df["us_domestic"] & ~df["is_lifestyle"]]
    focus = core[core["countries_focus"].map(len) > 0].explode("countries_focus")
    focus["weight"] = 1 / focus["thread_n"].clip(lower=1) ** 0.5

    # Баланс реакції на Україну за місяцями (середнє 1/0/−1 = частка позитиву − негативу)
    df.groupby("month")["reaction_ukraine"].mean()

    # Тип підтримки окремо для кожної країни: {"Germany": ["aid_military"], "Poland": ["refugees"]}
    import json
    cs = df["country_support"].fillna("{}").map(json.loads)
"""

from __future__ import annotations

import csv
import html
import json
import os
import random
import re
import shutil
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # without tqdm, plain prints
    tqdm = None


# ============================================================
# 0. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "output"          # parser output
INPUT_GLOB = "reddit_*.csv"

OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_CSV = OUTPUT_DIR / "reddit_ukraine_support.csv"

SAVE_PREVIEW = True                      # N random rows per month to look through
PREVIEW_FILE = OUTPUT_DIR / "preview_by_month.csv"
PREVIEW_PER_MONTH = 40                   # 24 months x 40 = ~1000 rows

# Technical columns (author, thread_title, permalink, country_context etc.) are not needed
# for the analysis and only make the file bigger. True keeps them for checking the filter.
SAVE_DEBUG_COLUMNS = False

SAVE_VALIDATION_SAMPLE = True            # sample for checking the filter by hand
VALIDATION_FILE = OUTPUT_DIR / "validation_sample.csv"
VALIDATION_PER_PERIOD = 100              # kept and dropped rows per period

TMP_DIR = OUTPUT_DIR / "_tmp_parts"
CHUNK_SIZE = 200_000
MIN_WORDS = 3                            # "Thank you Poland" should pass
SENTENCE_MAX_CHARS = 300                 # max length of one context sentence
CONTEXT_MAX_CHARS = 600                  # max context per country
THREAD_TITLE_MAX_CHARS = 150
RANDOM_SEED = 42
GAP_MIN_MONTHLY = 100                    # threshold for coverage_gaps()

# Same months in both years to avoid seasonality
PERIOD_WINDOWS = [
    ("2019-01-01", "2019-12-31"),
    ("2022-01-01", "2022-12-31"),
]
INVASION_DATE = "2022-02-24"

# Subreddits where everything counts as on topic (None = always, otherwise only in these periods).
# r/ukraine in 2019 is mostly travel, language and daily life, so only after the invasion.
DEDICATED_SUBREDDITS = {
    "ukrainianconflict": None,
    "ukrainewarreport": None,
    "ukraine": {"2022_war"},
}

WANTED_COLUMNS = [
    "id", "type", "subreddit", "datetime", "created_utc", "author", "author_flair",
    "score", "num_comments", "upvote_ratio", "title", "selftext", "body", "text",
    "link_id", "parent_id", "permalink",
]


# ============================================================
# 1. LEXICONS
# ============================================================
# Terms are lowercase. A space matches any whitespace, a hyphen matches hyphen/space/nothing
# ("lend-lease" also catches "lend lease"). "re:" prefix = raw regex, non-capturing groups only.

UKRAINE_TERMS = [
    "ukraine", "ukrainian", "ukrainians", "ukraina", "kyiv", "kiev",
    "zelensky", "zelenskyy", "zelenskiy", "zelenskyi", "zelenski",
    "verkhovna rada", "zsu", "crimea", "crimean", "donbas", "donbass",
    "donetsk", "luhansk", "lugansk", "dpr", "lpr",
    "kharkiv", "kharkov", "kherson", "mariupol", "zaporizhzhia", "zaporizhia",
    "zaporozhye", "zaporizhzhya", "odesa", "odessa", "dnipro", "dnipropetrovsk",
    "bakhmut", "avdiivka", "bucha", "irpin", "hostomel", "melitopol", "izium", "izyum",
    "sievierodonetsk", "severodonetsk", "lysychansk", "chernihiv", "sumy",
    "mykolaiv", "nikolaev", "lviv", "lvov", "snake island", "azovstal", "azov", "kerch",
    "poroshenko", "yanukovych", "maidan", "euromaidan", "naftogaz", "burisma",
    "slava ukraini",
    # flags are handled by FLAG_RE, otherwise "🇫🇷🇺🇦" would read as 🇷🇺
    # no "chernobyl": the HBO series (May 2019) would bring lots of unrelated posts
]

RUSSIA_TERMS = [
    "russia", "russian", "russians", "russian federation", "putin", "putler",
    "kremlin", "moscow", "lavrov", "shoigu", "peskov", "medvedev", "prigozhin",
    "wagner group", "kadyrov", "kadyrovites", "gerasimov", "gazprom", "rosneft",
    "ruzzia", "ruzzian", "ruzzians", "rashist", "rashists",
]

# War vocabulary (category "war")
WAR_CONTEXT_TERMS = [
    "war", "wars", "invasion", "invade", "invades", "invaded", "invading", "invader",
    "invaders", "aggression", "aggressor", "occupation", "occupied", "occupier",
    "occupiers", "annex", "annexed", "annexation", "troops", "soldiers", "army",
    "military", "frontline", "front line", "offensive", "counteroffensive",
    "counter-offensive", "shelling", "shelled", "missile", "missiles", "artillery",
    "tank", "tanks", "drone", "drones", "airstrike", "airstrikes", "bombing", "bombed",
    "mobilization", "mobilisation", "conscription", "sanction", "sanctions",
    "sanctioned", "embargo", "ceasefire", "peace talks", "war crimes", "genocide",
    "refugees", "nato", "weapons", "ammunition", "casualties", "border",
]

# Aggression markers. A text about Russia without Ukraine is on topic only if it has one of these.
# Generic words (military, missile, troops, war...) don't work: in 2019 they pulled in the INF
# treaty, Iran, Japan and even "polar bears invade Russian town" (hence no "invade" either).
RUSSIA_AGGRESSION_TERMS = [
    "invasion", "invasions", "invades", "invaded", "invading", "invader", "invaders",
    "aggression", "aggressor", "war of aggression", "russian aggression",
    "russian invasion", "russian occupation", "putin's war", "russia's war",
    "annex", "annexed", "annexation", "annexing",
    "occupation", "occupied", "occupier", "occupiers", "occupying",
    "mobilization", "mobilisation", "mobilize", "mobilise", "mobilized", "mobilised",
    "special military operation", "war crimes", "war criminal", "hybrid war",
    "frontline", "front line", "front lines", "counteroffensive", "counter-offensive",
    "shelling", "shelled", "ceasefire", "cease-fire", "peace talks",
    "sanction", "sanctions", "sanctioned", "embargo",
    "nato expansion", "nato enlargement", "eastern flank",
    "russian troops", "russian forces", "russian army", "russian soldiers",
    "military buildup", "troop buildup",
]

# Off-topic subjects for texts about Russia without Ukraine: other conflicts, Mueller and
# Trump in 2019, the Skripals, Navalny.
RUSSIA_OFFTOPIC_TERMS = [
    "syria", "syrian", "assad", "idlib", "aleppo", "mueller", "collusion",
    "russiagate", "election interference", "election meddling", "russian interference",
    "russian meddling", "steele dossier", "manafort", "venezuela", "maduro",
    "libya", "haftar", "navalny", "central african republic", "mali",
    "trump", "deripaska", "rusal", "flynn", "kompromat", "mcconnell", "moscow mitch",
    "mnuchin", "skripal", "novichok", "salisbury", "golan", "kuril", "kurils", "kurile",
]

# US domestic politics around Ukraine (2019 impeachment, Burisma, Hunter Biden).
# Not removed, since the scandal was about military aid, only flagged.
US_DOMESTIC_TERMS = [
    "impeach", "impeached", "impeachment", "impeaching", "whistleblower",
    "quid pro quo", "burisma", "hunter biden", "giuliani", "rudy", "sondland",
    "kurt volker", "vindman", "perfect call", "hunter's laptop", "hunter laptop",
    "call with zelensky", "zelensky call", r"re:trump(?:'s)?\s+(?:phone\s+)?call",
    "schiff", "adam schiff", "nadler", "nunes", "mulvaney", "parnas", "lev parnas",
    "fruman", "shokin", "lutsenko", "crowdstrike", "yovanovitch", "subpoena",
    "subpoenas", "subpoenaed", "whistle-blower", "whistleblowers", "call transcript",
    "transcript of the call", "impeachment inquiry", "articles of impeachment",
]

# In 2019 Trump + Ukraine almost always means the impeachment, so these are us_domestic too
US_SCANDAL_PERIODS = {"2019"}
TRUMP_RE = re.compile(r"(?<!\w)trump(?:'s)?(?!\w)", re.IGNORECASE)

# Lifestyle topics (travel, dating, food), only the is_lifestyle flag
LIFESTYLE_TERMS = [
    "travel", "travelling", "traveling", "trip", "tourist", "tourists", "tourism",
    "vacation", "dating", "girlfriend", "boyfriend", "wedding", "marry", "apartment",
    "restaurant", "restaurants", "recipe", "recipes", "cooking", "borscht", "borsch",
    "cuisine", "learn ukrainian", "learning ukrainian", "learn russian",
    "learning russian", "duolingo", "grammar", "football", "soccer", "salary",
    "moving to",
]

# "send ... weapons" with a few words in between
_SUPPLY_VERBS = (
    r"(?:send|sends|sending|sent|supply|supplies|supplying|supplied|deliver|delivers|"
    r"delivering|delivered|provide|provides|providing|provided|give|gives|giving|gave|"
    r"donate|donates|donating|donated|transfer|transfers|transferring|transferred|"
    r"ship|ships|shipping|shipped|pledge|pledges|pledged)"
)
_WEAPON_NOUNS = (
    r"(?:weapons?|arms|tanks?|artillery|howitzers?|missiles?|rockets?|jets?|planes?|"
    r"aircraft|drones?|ammunition|ammo|shells|air\s+defen[cs]e|anti-tank|"
    r"anti-aircraft|armou?red\s+vehicles|apcs|ifvs|mlrs)"
)

SUPPORT_TERMS: dict[str, list[str]] = {
    "aid_military": [
        "military aid", "military assistance", "military support", "security assistance",
        "lethal aid", "defense package", "defence package", "weapons package",
        "arms package", "arms supplies", "arms deliveries", "arms shipments",
        "arms shipment", "weapons deliveries", "weapons shipments", "ammunition",
        "himars", "m142", "m270", "m777", "m777s", "javelin", "javelins", "nlaw", "nlaws",
        "stinger", "stingers", "starstreak", "switchblade", "switchblades", "phoenix ghost",
        "bayraktar", "bayraktars", "tb2", "leopard 2", "leopards", "abrams",
        "challenger 2", "marder", "marders", "bradley", "bradleys", "gepard", "gepards",
        "iris-t", "nasams", "patriot missiles", "patriot systems", "caesar howitzer",
        "caesar howitzers", "krab", "pzh 2000", "piorun", "bushmaster", "bushmasters",
        "lend-lease", "ramstein", "military training", "training ukrainian",
        "intelligence sharing", "no-fly zone", "fighter jets", "mig-29s", "f-16s",
        f"re:{_SUPPLY_VERBS}(?:\\W+\\w+){{0,4}}?\\W+{_WEAPON_NOUNS}",
    ],
    "aid_financial": [
        "financial aid", "financial assistance", "financial support", "economic aid",
        "economic assistance", "budget support", "macro-financial", "macrofinancial",
        "loan", "loans", "grant", "grants", "funding", "funds", "fund", "donation",
        "donations", "donate", "donated", "reconstruction", "rebuild ukraine",
        "marshall plan", "imf", "world bank", "debt relief",
        "re:\\$\\s?\\d+(?:[.,]\\d+)?\\s*(?:b|bn|billion|m|mn|million)",
        "re:\\d+(?:[.,]\\d+)?\\s*(?:billion|bn)",
    ],
    "aid_humanitarian": [
        "humanitarian", "medical supplies", "medicine", "medicines", "medical aid",
        "food aid", "generators", "red cross", "unicef", "world central kitchen",
        "shelter", "shelters", "evacuate", "evacuated", "evacuation", "evacuations",
        "volunteers", "volunteer", "charity", "charities", "humanitarian corridor",
    ],
    "refugees": [
        "refugee", "refugees", "asylum", "displaced", "fleeing", "fled",
        "temporary protection", "border crossing", "host refugees", "hosting refugees",
        "housing refugees", "welcomed ukrainians", "welcoming ukrainians",
    ],
    "sanctions": [
        "sanction", "sanctions", "sanctioned", "sanctioning", "embargo", "embargoes",
        "swift", "price cap", "frozen assets", "freeze assets", "asset freeze",
        "seized assets", "seize assets", "oligarch", "oligarchs", "export controls",
        "export ban", "import ban", "ban on russian", "boycott", "boycotting",
        "pull out of russia", "pulled out of russia", "leave russia", "leaving russia",
        "exit russia", "exiting russia", "closed airspace", "close airspace",
    ],
    "energy": [
        "gas", "natural gas", "lng", "oil", "crude", "pipeline", "pipelines",
        "nord stream", "nordstream", "gazprom", "energy", "energy crisis",
        "energy prices", "electricity", "heating", "coal", "gas prices", "oil prices",
        "energy dependence", "energy independence", "refinery",
    ],
    "diplomatic": [
        "diplomatic", "diplomacy", "diplomat", "diplomats", "ambassador", "ambassadors",
        "embassy", "embassies", "summit", "condemn", "condemns", "condemned",
        "condemnation", "resolution", "security council", "general assembly",
        "negotiation", "negotiations", "negotiate", "peace talks", "ceasefire",
        "peace deal", "mediator", "mediate", "mediation", "recognize", "recognise",
        "recognition", "visited kyiv", "visit to kyiv", "visits kyiv", "normandy format",
        "minsk agreements", "minsk agreement", "minsk accords", "g7", "g20", "osce",
        "expelled", "expel", "persona non grata", "solidarity", "stand with ukraine",
        "stands with ukraine", "standing with ukraine",
    ],
    "membership": [
        "join nato", "joining nato", "joined nato", "nato membership", "nato accession",
        "membership", "accession", "candidate status", "candidacy", "eu candidate",
        "join the eu", "joining the eu", "eu membership", "nato expansion",
        "nato enlargement", "eu enlargement", "article 5", "open door policy",
    ],
}
SUPPORT_SHORT = {
    "aid_military": "military", "aid_financial": "financial",
    "aid_humanitarian": "humanitarian", "refugees": "refugees",
    "sanctions": "sanctions", "energy": "energy",
    "diplomatic": "diplomatic", "membership": "membership",
}

# ------------------------------------------------------------
# Countries and organizations
# terms - case-insensitive, cs - case-sensitive
# nat - nationality ("I'm American", "as a Pole")
# place - name for "here in Poland", "I'm from Germany" and flairs
# iso - emoji flag code
# Ambiguous words (english, parliament, congress, washington, johnson, tusk, rome,
# hague, baltic...) are removed or limited by context.
# ------------------------------------------------------------
_NOT_AMERICAS = r"(?<!south )(?<!latin )(?<!north )(?<!central )(?<!native )"
_CAPS_GUARD = r"(?<![A-Z]{2}\s)"   # "KILL US" in caps is not the USA
# Austria-Hungary counts neither as Hungary nor as Austria
_NOT_AUSTRIA_HUNGARY = (r"(?<!austria-)(?<!austria–)(?<!austria/)(?<!austria )(?<!austria - )"
                        r"(?<!austria – )(?<!austro-)(?<!austro–)(?<!austro )")
# CAESAR (howitzer) only in caps or next to artillery words,
# since Caesar is usually Julius Caesar or a game character
_CAESAR_WEAPON = r"caesars?(?=\s+(?:howitzers?|spgs?|self[-\s]?propelled|artillery|guns?|systems?))"

ACTORS: dict[str, dict[str, Any]] = {
    "USA": dict(kind="country", iso="US",
        terms=["united states", "united states of america", "usa", "u.s.a.", "u.s.a",
               "u.s.", "u.s", f"re:{_NOT_AMERICAS}americans?", f"re:{_NOT_AMERICAS}america",
               "biden", "joe biden", "blinken", "lloyd austin", "kamala harris", "pelosi",
               "nuland", "trump", "donald trump", "pompeo", "pence", "mark esper",
               "white house", "pentagon", "state department", "department of defense",
               "capitol hill", "cia", "lend-lease", "himars", "javelin", "javelins",
               "stinger", "stingers", "m777", "m777s", "abrams", "switchblade",
               "switchblades", "phoenix ghost", "patriot missiles", "patriot systems"],
        cs=[f"re:{_CAPS_GUARD}US"],
        nat=["american"],
        place=["united states", "usa", "u.s.", "u.s", "us", "america", "states"]),
    "UK": dict(kind="country", iso="GB",
        terms=["united kingdom", "uk", "u.k.", "u.k", "britain", "great britain",
               "british", "brits", "england", "scotland", "wales", "boris johnson", "bojo",
               "liz truss", "rishi sunak", "sunak", "ben wallace", "theresa may",
               "dominic raab", "james cleverly", "downing street", "westminster",
               "whitehall", "london", "mi6", "royal navy", "royal air force", "raf",
               "british army", "nlaw", "nlaws", "starstreak", "brimstone",
               "challenger 2", "challenger 2s"],
        nat=["british", "brit", "english", "scottish", "welsh", "scot"],
        place=["uk", "u.k.", "united kingdom", "britain", "great britain", "england",
               "scotland", "wales"]),
    "Poland": dict(kind="country", iso="PL",
        terms=["poland", r"re:warsaw(?!\s+pact)", "warszawa", "andrzej duda", "duda",
               "morawiecki",
               "kaczynski", "kaczyński", "rzeszow", "rzeszów", "przemysl", "przemyśl",
               "piorun", "krab"],
        cs=["Polish", "Poles"],
        nat=["polish", "pole"], place=["poland"]),
    "Germany": dict(kind="country", iso="DE",
        terms=["germany", "german", "germans", "deutschland", "scholz", "olaf scholz",
               "merkel", "angela merkel", "baerbock", "steinmeier", "lambrecht", "habeck",
               "lindner", "heiko maas", "berlin", "bundestag", "bundesrat", "bundeswehr",
               "leopard", "leopards", "leopard 2", "leopard 2s", "gepard", "gepards",
               "iris-t", "pzh 2000", "panzerhaubitze", "marder", "marders", "rheinmetall"],
        nat=["german"], place=["germany", "deutschland"]),
    "France": dict(kind="country", iso="FR",
        terms=["france", "french", "macron", "emmanuel macron", "elysee", "élysée",
               "paris", "le drian", "colonna", "le pen", "marine le pen", "zemmour",
               "caesar howitzer", "caesar howitzers", f"re:{_CAESAR_WEAPON}"],
        cs=["CAESAR", "CAESARs"],
        nat=["french", "frenchman", "frenchwoman"], place=["france"]),
    "Hungary": dict(kind="country", iso="HU",
        terms=[f"re:{_NOT_AUSTRIA_HUNGARY}hungar(?:y|ians?)",
               "orban", "orbán", "viktor orban", "viktor orbán", "szijjarto", "szijjártó",
               "fidesz",
               r"re:budapest(?!\s+memorandum)"],   # not the Budapest Memorandum
        nat=["hungarian"], place=["hungary"]),
    "Turkey": dict(kind="country", iso="TR",
        terms=["turkish", "turkiye", "türkiye", "erdogan", "erdoğan", "ankara",
               "istanbul", "bosphorus", "bosporus", "dardanelles", "montreux convention",
               "cavusoglu", "çavuşoğlu", "bayraktar", "bayraktars", "tb2", "tb2s"],
        cs=["Turkey", "Turks"],                    # lowercase "turkey" is the bird
        nat=["turkish", "turk"], place=["turkey", "turkiye", "türkiye"]),
    "Canada": dict(kind="country", iso="CA",
        terms=["canada", "canadian", "canadians", "trudeau", "justin trudeau", "ottawa",
               "freeland", "chrystia freeland", "anita anand", "melanie joly",
               "mélanie joly"],
        nat=["canadian"], place=["canada"]),
    "China": dict(kind="country", iso="CN",
        terms=["china", "chinese", "beijing", "xi jinping", "ccp", "prc",
               "people's republic of china", "communist party of china", "wang yi"],
        cs=["Xi"],
        nat=["chinese"], place=["china"]),
    "Czechia": dict(kind="country", iso="CZ",
        terms=["czechia", "czech republic", "czech", "czechs", "prague", "petr fiala",
               "fiala", "babis", "babiš", "zeman", "lipavsky", "lipavský"],
        nat=["czech"], place=["czechia", "czech republic"]),
    "Slovakia": dict(kind="country", iso="SK",
        terms=["slovakia", "slovak", "slovaks", "slovakian", "bratislava", "heger",
               "eduard heger", "caputova", "čaputová", "fico", "robert fico"],
        nat=["slovak", "slovakian"], place=["slovakia"]),
    "Romania": dict(kind="country", iso="RO",
        terms=["romania", "romanian", "romanians", "bucharest", "iohannis",
               "klaus iohannis", "ciuca", "ciucă"],
        nat=["romanian"], place=["romania"]),
    "Lithuania": dict(kind="country", iso="LT",
        terms=["lithuania", "lithuanian", "lithuanians", "vilnius", "nauseda", "nausėda",
               "landsbergis", "seimas"],
        nat=["lithuanian"], place=["lithuania"]),
    "Latvia": dict(kind="country", iso="LV",
        terms=["latvia", "latvian", "latvians", "riga", "karins", "kariņš", "saeima"],
        nat=["latvian"], place=["latvia"]),
    "Estonia": dict(kind="country", iso="EE",
        terms=["estonia", "estonian", "estonians", "tallinn", "kaja kallas", "kallas",
               "alar karis", "riigikogu"],
        nat=["estonian"], place=["estonia"]),
    "Baltics": dict(kind="country", iso=None,
        terms=["baltic states", "baltic countries", "baltic nations", "baltic republics",
               "baltics"]),                        # plain "baltic" can mean the sea
    "Italy": dict(kind="country", iso="IT",
        terms=["italy", "italian", "italians", "draghi", "mario draghi", "meloni",
               "giorgia meloni", "salvini", "berlusconi", "di maio"],
        nat=["italian"], place=["italy"]),
    "Netherlands": dict(kind="country", iso="NL",
        terms=["netherlands", "the netherlands", "dutch", "holland", "rutte",
               "mark rutte", "amsterdam", "hoekstra", "wopke hoekstra"],
        nat=["dutch"], place=["netherlands", "the netherlands", "holland"]),
    "Sweden": dict(kind="country", iso="SE",
        terms=["sweden", "swedish", "stockholm", "magdalena andersson",
               "ulf kristersson", "kristersson"],
        cs=["Swedes"],
        nat=["swedish", "swede"], place=["sweden"]),
    "Finland": dict(kind="country", iso="FI",
        terms=["finland", "finnish", "helsinki", "niinisto", "niinistö", "sauli niinisto",
               "sauli niinistö", "sanna marin"],
        cs=["Finns"],
        nat=["finnish", "finn"], place=["finland"]),
    "Norway": dict(kind="country", iso="NO",
        terms=["norway", "norwegian", "norwegians", "oslo", "jonas gahr store",
               "jonas gahr støre", "gahr store", "gahr støre", "erna solberg", "equinor"],
        nat=["norwegian"], place=["norway"]),
    "Denmark": dict(kind="country", iso="DK",
        terms=["denmark", "danish", "copenhagen", "mette frederiksen", "frederiksen"],
        cs=["Danes"],
        nat=["danish", "dane"], place=["denmark"]),
    "Spain": dict(kind="country", iso="ES",
        terms=["spain", "spanish", "spaniards", "pedro sanchez", "pedro sánchez"],
        nat=["spanish", "spaniard"], place=["spain"]),
    "Belgium": dict(kind="country", iso="BE",
        terms=["belgium", "belgian", "belgians", "alexander de croo", "de croo"],
        nat=["belgian"], place=["belgium"]),
    "Austria": dict(kind="country", iso="AT",
        terms=[r"re:austria(?!\s*[-–/]\s*hungar|\s+hungar)", "austrian", "austrians",
               "vienna", "nehammer", "karl nehammer"],
        nat=["austrian"], place=["austria"]),
    "Switzerland": dict(kind="country", iso="CH",
        terms=["switzerland", "swiss", "ignazio cassis"],
        nat=["swiss"], place=["switzerland"]),
    "Greece": dict(kind="country", iso="GR",
        terms=["greece", "greek", "greeks", "athens", "mitsotakis"],
        nat=["greek"], place=["greece"]),
    "Bulgaria": dict(kind="country", iso="BG",
        terms=["bulgaria", "bulgarian", "bulgarians", "radev", "rumen radev",
               "kiril petkov", "petkov"],
        nat=["bulgarian"], place=["bulgaria"]),
    "Portugal": dict(kind="country", iso="PT",
        terms=["portugal", "portuguese"],
        nat=["portuguese"], place=["portugal"]),
    "Serbia": dict(kind="country", iso="RS",
        terms=["serbia", "serbian", "serbians", "serbs", "belgrade", "vucic", "vučić",
               "aleksandar vucic"],
        nat=["serbian", "serb"], place=["serbia"]),
    "Moldova": dict(kind="country", iso="MD",
        terms=["moldova", "moldovan", "moldovans", "chisinau", "chișinău", "chişinău",
               "transnistria", "transnistrian", "maia sandu"],
        nat=["moldovan"], place=["moldova"]),
    "Belarus": dict(kind="country", iso="BY",
        terms=["belarus", "belarusian", "belarusians", "belorussia", "byelorussia",
               "lukashenko", "lukashenka", "tsikhanouskaya", "tikhanovskaya",
               # Minsk agreements are not Belarus
               r"re:minsk(?!\s+(?:agreements?|accords?|protocols?|ii|2|deal|format|process))"],
        nat=["belarusian"], place=["belarus"]),
    "Iran": dict(kind="country", iso="IR",
        terms=["iran", "iranian", "iranians", "tehran", "khamenei", "raisi",
               "ebrahim raisi", "shahed", "shaheds", "shahed-136", "mohajer"],
        nat=["iranian"], place=["iran"]),
    "Israel": dict(kind="country", iso="IL",
        terms=["israel", "israeli", "israelis", "tel aviv", "naftali bennett",
               "yair lapid", "lapid", "iron dome", "mossad"],
        nat=["israeli"], place=["israel"]),
    "India": dict(kind="country", iso="IN",
        terms=["india", "new delhi", "delhi", "modi", "narendra modi", "jaishankar",
               "indian government"],
        nat=["indian"], place=["india"]),
    "Japan": dict(kind="country", iso="JP",
        terms=["japan", "japanese", "tokyo", "kishida", "fumio kishida", "shinzo abe"],
        nat=["japanese"], place=["japan"]),
    "South_Korea": dict(kind="country", iso="KR",
        terms=["south korea", "south korean", "south koreans", "republic of korea",
               "seoul", "yoon suk yeol", "moon jae-in"],
        nat=["south korean", "korean"], place=["south korea", "korea"]),
    "North_Korea": dict(kind="country", iso="KP",
        terms=["north korea", "north korean", "north koreans", "dprk", "pyongyang",
               "kim jong un", "kim jong-un"],
        nat=["north korean"], place=["north korea"]),
    "Australia": dict(kind="country", iso="AU",
        terms=["australia", "australian", "australians", "canberra", "scott morrison",
               "anthony albanese", "bushmaster", "bushmasters"],
        nat=["australian", "aussie"], place=["australia"]),
    # --- Organizations ---
    "NATO": dict(kind="org", iso=None,
        terms=["nato", "north atlantic treaty organization",
               "north atlantic treaty organisation", "north atlantic alliance", "otan",
               "stoltenberg", "jens stoltenberg", "article 5", "madrid summit"]),
    "EU": dict(kind="org", iso="EU",
        terms=["european union", "e.u.", "e.u", "european commission",
               "european council", "european parliament", "von der leyen",
               "ursula von der leyen", "borrell", "josep borrell", "charles michel",
               "brussels"],
        cs=[f"re:{_CAPS_GUARD}EU"]),
    "UN": dict(kind="org", iso="UN",
        terms=["united nations", "un security council", "unsc", "un general assembly",
               "unga", "guterres", "antonio guterres", "unhcr", "unicef", "iaea",
               "un human rights"],
        cs=[f"re:{_CAPS_GUARD}UN"]),
}

# author country can also be Ukraine or Russia
SELF_ONLY: dict[str, dict[str, Any]] = {
    "Ukraine": dict(iso="UA", nat=["ukrainian"], place=["ukraine"]),
    "Russia": dict(iso="RU", nat=["russian"], place=["russia"]),
}

BOT_AUTHORS = {
    "automoderator", "autotldr", "remindmebot", "wikitextbot", "sneakpeekbot",
    "converter-bot", "timezone_bot", "savevideobot", "repostsleuthbot",
    "translator-bot", "goodbot_badbot", "b0trank", "auto-xkcd37", "vredditshare",
    "gifreversingbot", "wikimobilelinkbot", "totesmessenger", "imguralbumbot",
}


# ============================================================
# 2. REGEXES AND TERM LEXICON
# ============================================================
# One pass with a regex trie over all terms instead of ~15 separate ones; LEXICON then
# tells which categories and countries a matched term belongs to.

def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def term_key(s: str) -> str:
    """Term key: "Lend-Lease" -> "lendlease"."""
    return re.sub(r"[\s\-]+", "", s.lower())


def term_to_regex(term: str) -> str:
    if term.startswith("re:"):
        return term[3:]
    parts = []
    for p in term.strip().split():
        e = re.escape(p).replace(r"\-", "-").replace("-", r"[-\s]?")
        parts.append(e)
    return r"\s+".join(parts)


def trie_regex(terms) -> str:
    """Word list to a regex trie: "german(?:s|y)?" instead of "germans|germany|german"."""
    trie: dict = {}
    for t in terms:
        node = trie
        for ch in " ".join(t.lower().split()):
            node = node.setdefault(ch, {})
        node[""] = {}

    def char_re(ch: str) -> str:
        if ch == " ":
            return r"\s+"
        if ch == "-":
            return r"[-\s]?"
        return re.escape(ch)

    def emit(node: dict) -> str:
        branches = [char_re(ch) + emit(child) for ch, child in sorted(node.items()) if ch]
        if not branches:
            return ""
        body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
        return "(?:" + body + ")?" if "" in node else body

    return emit(trie)


def alternation(terms) -> str:
    lits = [t for t in terms if not t.startswith("re:")]
    raws = [t[3:] for t in terms if t.startswith("re:")]
    return "|".join(([trie_regex(lits)] if lits else []) + raws)


# noinspection RegExpUnnecessaryNonCapturingGroup
def compile_terms(terms) -> re.Pattern:
    return re.compile(r"(?<!\w)(?:" + alternation(terms) + r")(?!\w)", re.IGNORECASE)


CATEGORY_TERMS: dict[str, list[str]] = {
    "ukraine": UKRAINE_TERMS,
    "russia": RUSSIA_TERMS,
    "war": WAR_CONTEXT_TERMS,
    "aggression": RUSSIA_AGGRESSION_TERMS,
    "offtopic": RUSSIA_OFFTOPIC_TERMS,
    "us_domestic": US_DOMESTIC_TERMS,
    "lifestyle": LIFESTYLE_TERMS,
    **SUPPORT_TERMS,
}
SUPPORT_CATS: frozenset[str] = frozenset(SUPPORT_TERMS)
UKR_RUS_RE = compile_terms(UKRAINE_TERMS + RUSSIA_TERMS + ["🇺🇦", "🇷🇺"])  # fast prefilter
EMPTY: frozenset[str] = frozenset()


def cfg_list(cfg: dict[str, Any], key: str) -> list[str]:
    """Country terms / cs / nat / place as a list of strings."""
    value = cfg.get(key) or []
    return [str(item) for item in value]


def cfg_str(cfg: dict[str, Any], key: str) -> str:
    """Country kind / iso as a string."""
    value = cfg.get(key)
    return str(value) if value else ""


def actor_literal_terms() -> list[str]:
    """All country terms except the "re:" ones."""
    result: list[str] = []
    for cfg in ACTORS.values():
        result.extend(term for term in cfg_list(cfg, "terms") if not term.startswith("re:"))
    return result


def build_actor_index() -> tuple[dict[str, str], list[str]]:
    """Term key -> country, plus separate regex groups for cs and "re:" terms."""
    term_to_actor: dict[str, str] = {}
    special_groups: list[str] = []
    for key, cfg in ACTORS.items():
        terms = cfg_list(cfg, "terms")
        for term in terms:
            if term.startswith("re:"):
                continue
            tk = term_key(term)
            if term_to_actor.get(tk, key) != key:
                raise ValueError(f"Term '{term}' belongs to two countries: "
                                 f"{term_to_actor[tk]} and {key}")
            term_to_actor[tk] = key
        spec: list[str] = [term[3:] for term in terms if term.startswith("re:")]
        spec += [f"(?-i:{term_to_regex(term)})" for term in cfg_list(cfg, "cs")]
        if spec:
            special_groups.append(f"(?P<{key}>{'|'.join(spec)})")
    return term_to_actor, special_groups


def build_category_res() -> dict[str, re.Pattern]:
    result: dict[str, re.Pattern] = {}
    for category, terms in CATEGORY_TERMS.items():
        result[category] = compile_terms(terms)
    return result


def build_lexicon() -> tuple[dict[str, tuple[frozenset[str], frozenset[str]]], set[str]]:
    """
    Term key -> (categories, countries). A long term also gets the categories of words in it:
    "nato membership" gives membership, war and NATO.
    """
    literals: set[str] = set()
    for terms in CATEGORY_TERMS.values():
        literals.update(term for term in terms if not term.startswith("re:"))
    literals.update(actor_literal_terms())

    lexicon: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for term in literals:
        cats = frozenset(c for c, rx in CATEGORY_RES.items() if rx.search(term))
        acts_set: set[str] = set()
        for m in ACTOR_LIT_RE.finditer(term):
            found_key = term_key(m.group(0))
            if found_key in TERM_TO_ACTOR:
                acts_set.add(TERM_TO_ACTOR[found_key])
        tk = term_key(term)
        old_cats, old_acts = lexicon.get(tk, (EMPTY, EMPTY))
        lexicon[tk] = (old_cats | cats, old_acts | frozenset(acts_set))
    return lexicon, literals


def build_category_special() -> list[str]:
    """Regex terms of categories ("send ... weapons", "$40 billion")."""
    groups: list[str] = []
    for category, terms in CATEGORY_TERMS.items():
        raws = [term[3:] for term in terms if term.startswith("re:")]
        if raws:
            groups.append(f"(?P<{category}>{'|'.join(raws)})")
    return groups


def build_self_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Nationality and place -> country ("As an American", "here in Poland")."""
    nat_map: dict[str, str] = {}
    place_map: dict[str, str] = {}
    for key, cfg in ACTORS.items():
        if cfg_str(cfg, "kind") != "country":
            continue
        for word in cfg_list(cfg, "nat"):
            nat_map[norm(word)] = ACTOR_NAME[key]
        for word in cfg_list(cfg, "place"):
            place_map[norm(word)] = ACTOR_NAME[key]
    for name, cfg in SELF_ONLY.items():
        for word in cfg_list(cfg, "nat"):
            nat_map[norm(word)] = name
        for word in cfg_list(cfg, "place"):
            place_map[norm(word)] = name
    return nat_map, place_map


def build_iso_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Flag ISO code -> actor key and country name (for flairs)."""
    iso_to_actor: dict[str, str] = {}
    iso_to_country: dict[str, str] = {}
    for key, cfg in ACTORS.items():
        iso = cfg_str(cfg, "iso")
        if not iso:
            continue
        iso_to_actor[iso] = key
        if cfg_str(cfg, "kind") == "country":
            iso_to_country[iso] = ACTOR_NAME[key]
    for name, cfg in SELF_ONLY.items():
        iso = cfg_str(cfg, "iso")
        if iso:
            iso_to_country[iso] = name
    return iso_to_actor, iso_to_country


CATEGORY_RES = build_category_res()

ACTOR_NAME: dict[str, str] = {key: key.replace("_", " ") for key in ACTORS}
ACTOR_KIND: dict[str, str] = {key: cfg_str(cfg, "kind") for key, cfg in ACTORS.items()}
TERM_TO_ACTOR, _ACTOR_SPECIAL_GROUPS = build_actor_index()
# noinspection RegExpUnnecessaryNonCapturingGroup
ACTOR_LIT_RE = re.compile(r"(?<!\w)(?:" + trie_regex(actor_literal_terms()) + r")(?!\w)",
                          re.IGNORECASE)
# noinspection RegExpUnnecessaryNonCapturingGroup
ACTOR_SPECIAL_RE = re.compile(r"(?<!\w)(?:" + "|".join(_ACTOR_SPECIAL_GROUPS) + r")(?!\w)",
                              re.IGNORECASE)

LEXICON, _LITERALS = build_lexicon()
# noinspection RegExpUnnecessaryNonCapturingGroup
LEX_RE = re.compile(r"(?<!\w)(?:" + trie_regex(_LITERALS) + r")(?!\w)", re.IGNORECASE)
# noinspection RegExpUnnecessaryNonCapturingGroup
CAT_SPECIAL_RE = re.compile(r"(?<!\w)(?:" + "|".join(build_category_special()) + r")(?!\w)",
                            re.IGNORECASE)

# emoji flag = two regional indicator letters
FLAG_RE = re.compile("[\U0001F1E6-\U0001F1FF]{2}")


def flag_to_iso(flag: str) -> str:
    return "".join(chr(ord(ch) - 0x1F1E6 + ord("A")) for ch in flag)


ISO_TO_ACTOR, ISO_TO_COUNTRY = build_iso_maps()
FLAG_CATS: dict[str, frozenset[str]] = {"UA": frozenset(["ukraine"]), "RU": frozenset(["russia"])}

# author self-identification ("As an American", "here in Poland")
SELF_NAT, SELF_PLACE = build_self_maps()

# Sign-offs like "Love from the Netherlands" or "From Canada with love" give the author's
# country, not a target. "us" is skipped ("love from us all").
_GREETINGS = r"(?:love|greetings|regards|hugs|prayers|cheers|salute|best\s+wishes|hello|hi)"
_GREET_PLACES = alternation([p for p in SELF_PLACE if p != "us"])
_TRAILING_FLAGS = "(?:\\s*[\U0001F1E6-\U0001F1FF]{2})*"   # flag after the sign-off
SELF_ID_RE = re.compile(
    r"(?<!\w)(?:as\s+an?|i'?m\s+an?|i\s+am\s+an?|i'?m|i\s+am|fellow)\s+"
    r"(?P<nat>" + alternation(SELF_NAT) + r")s?(?!\w)"
    r"|(?<!\w)(?:here\s+in|i\s+live\s+in|i'?m\s+from|i\s+am\s+from|i\s+come\s+from|"
    r"living\s+in|born\s+in|back\s+home\s+in)\s+(?:the\s+)?"
    r"(?P<place>" + alternation(SELF_PLACE) + r")(?!\w)"
    r"|(?<!\w)" + _GREETINGS + r"\s+(?:\w+\s+)?from\s+(?:the\s+)?"
    r"(?P<gplace>" + _GREET_PLACES + r")(?!\w)" + _TRAILING_FLAGS +
    r"|(?<!\w)from\s+(?:the\s+)?(?P<wplace>" + _GREET_PLACES + r")\s+with\s+"
    r"(?:love|support|respect)(?!\w)" + _TRAILING_FLAGS,
    re.IGNORECASE,
)
SELF_HINT_RE = re.compile(r"(?<!\w)(?:i'?m|i\s+am|as\s+an?|fellow|here\s+in|live\s+in|"
                          r"living\s+in|born\s+in|from)(?!\w)", re.IGNORECASE)

_FLAIR_AMBIGUOUS = {"us", "states", "korea", "korean"}
SELF_ALL: dict[str, str] = {**SELF_PLACE, **SELF_NAT}
FLAIR_RE = compile_terms([word for word in SELF_ALL if word not in _FLAIR_AMBIGUOUS])

# Sentence boundaries; don't split on "U.S. Army" or "U. S."
SENT_SPLIT_RE = re.compile(
    r"(?<![A-Z]\.[A-Z]\.)(?<!\b[A-Z]\.)(?<=[.!?…])\s+"
    r"(?=[\"'(\[]?[A-Z0-9\U0001F1E6-\U0001F1FF])"
)


# ============================================================
# 3. TEXT CLEANING, LANGUAGE
# ============================================================

WS_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MD_LINK_RE = re.compile(r"\[([^]]*)]\([^)]*\)")
SUB_USER_RE = re.compile(r"(?<![\w/])/?[ru]/[A-Za-z0-9_-]+")
MD_CHARS_RE = re.compile(r"[*_~^`#|]+")
ENTITY_RE = re.compile(r"&(?:[a-z]+|#\d+|#x[0-9a-f]+);", re.IGNORECASE)
# ">" quote in text without line breaks: runs until a double space (former empty line)
QUOTE_FLAT_RE = re.compile(r"(?:^|(?<=\s\s))\s*>.*?(?=\s\s|$)")
BOT_TEXT_RE = re.compile(r"\bi(?:'m| am) a bot\b|performed automatically|\bbeep boop\b|"
                         r"^remindme!", re.IGNORECASE)
# moderator messages written by people, not AutoModerator
MOD_TEXT_RE = re.compile(
    r"has been removed (?:for|because|due)"
    r"|^,?\s*your (?:comment|submission|post)\b.{0,120}\bremoved"
    r"|^hi,? thank you for your contribution"
    r"|please (?:message|contact) the moderators",
    re.IGNORECASE)

WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
# rough language detection by function words
EN_STOPWORDS = frozenset("""
the a an and is are was were of to in on at by for with from that this it its it's
they you your have has had not but be been being what would should could can will just
don't doesn't didn't i'm their there they're about more because if or which who we our
he she his her them these those than then so do does did all any some my me how why
when where yes also only very too like get got thank thanks into out up no nobody now
here one as after before still even over against people
""".split())
OTHER_STOPWORDS = [
    frozenset("der die das und ist nicht ich sie es ein eine mit auf für den dem auch "
              "sich von zu wir aber wie noch nur schon wenn".split()),                # de
    frozenset("le la les et est des une un pas que qui pour dans sur avec ce il je "
              "nous vous mais sont du au".split()),                                    # fr
    frozenset("el los las y es que una por para con del pero como más muy está "
              "son".split()),                                                          # es
    frozenset("il che non una per con sono della del gli anche ma come questo "
              "è".split()),                                                            # it
    frozenset("i w z na nie się to jest że do co jak ale tak czy po od za przez "
              "jestem".split()),                                                       # pl
    frozenset("de het een en van niet dat op ik je zijn met voor maar ook er wat "
              "nog".split()),                                                          # nl
]


def clean_text(raw) -> str:
    """Decodes HTML, removes quotes, links, markdown, r/ and u/."""
    if not isinstance(raw, str) or not raw:
        return ""
    t = html.unescape(raw)
    if "&" in t and ENTITY_RE.search(t):          # double-escaped
        t = html.unescape(t)
    t = (t.replace("\r\n", "\n").replace("\r", "\n")
          .replace("\u200b", " ").replace("\xa0", " ")
          .replace("\u2019", "'").replace("\u2018", "'")
          .replace("\u201c", '"').replace("\u201d", '"'))
    if "\n" in t:                                  # line breaks kept: drop lines starting with ">"
        t = "\n".join(line for line in t.split("\n") if not line.lstrip().startswith(">"))
    elif ">" in t:                                 # line breaks already replaced by spaces
        t = QUOTE_FLAT_RE.sub(" ", t)
    t = MD_LINK_RE.sub(r"\1", t)
    t = URL_RE.sub(" ", t)
    t = SUB_USER_RE.sub(" ", t)
    t = MD_CHARS_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()


def text_stats(text: str) -> tuple[int, str]:
    """Word count and language: en / cyrillic / other."""
    words = WORD_RE.findall(text.lower())
    n = len(words)
    if n == 0:
        return 0, "other"
    cyr = len(CYRILLIC_RE.findall(text))
    if cyr and cyr * 2 > sum(len(w) for w in words):
        return n, "cyrillic"
    n_en = sum(1 for w in words if w in EN_STOPWORDS)
    n_other = max(sum(1 for w in words if w in sw) for sw in OTHER_STOPWORDS)
    if n_en >= 1 and n_en >= n_other:
        return n, "en"
    if n_en == 0 and n_other == 0 and n <= 6:      # "Slava Ukraini heroes" etc.
        return n, "en"
    return n, "other"


# ============================================================
# 4. TEXT ANALYSIS: CATEGORIES, COUNTRIES, CONTEXT, AUTHOR COUNTRY
# ============================================================

def scan(text: str):
    """Returns (categories, hits), hit = (start, end, categories, countries, term)."""
    cats: set[str] = set()
    hits = []
    for m in LEX_RE.finditer(text):
        c, a = LEXICON.get(term_key(m.group(0)), (EMPTY, EMPTY))
        if c or a:
            cats |= c
            hits.append((m.start(), m.end(), c, a, m.group(0)))
    for m in FLAG_RE.finditer(text):
        iso = flag_to_iso(m.group(0))
        if iso in FLAG_CATS:
            c, a = FLAG_CATS[iso], EMPTY
        elif iso in ISO_TO_ACTOR:
            c, a = EMPTY, frozenset([ISO_TO_ACTOR[iso]])
        else:
            continue
        cats |= c
        hits.append((m.start(), m.end(), c, a, "flag:" + iso))
    return cats, hits


def full_categories(text: str) -> set:
    cats, _ = scan(text)
    cats |= {str(m.lastgroup) for m in CAT_SPECIAL_RE.finditer(text) if m.lastgroup}
    return cats


def _sentence(text: str, starts: list[int], i: int, a: int, b: int) -> str:
    s = starts[i]
    e = starts[i + 1] if i + 1 < len(starts) else len(text)
    sent = text[s:e].strip()
    if len(sent) > SENTENCE_MAX_CHARS:
        half = SENTENCE_MAX_CHARS // 2
        sent = text[max(s, a - half):min(e, b + half)].strip()
    return sent


def _dump(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


def analyze_details(text: str, hits: list):
    """
    Returns (extra_cats, countries, organizations, terms_json, context_json, support_json,
    self_country, focus). focus = countries in the same sentence as Ukraine, support or
    Russia + aggression. "As an American" goes to the author country, not to mentions.
    The support type of a country comes from the sentences where it is mentioned.
    """
    hits = list(hits)
    extra: set[str] = set()
    for m in CAT_SPECIAL_RE.finditer(text):
        extra.add(str(m.lastgroup))
        hits.append((m.start(), m.end(), frozenset([m.lastgroup]), EMPTY, m.group(0)))
    for m in ACTOR_SPECIAL_RE.finditer(text):
        hits.append((m.start(), m.end(), EMPTY, frozenset([m.lastgroup]), m.group(0)))

    self_country, self_spans = None, []
    if SELF_HINT_RE.search(text):
        for m in SELF_ID_RE.finditer(text):
            if m.group("nat"):
                country = SELF_NAT.get(norm(m.group("nat")))
            else:
                place = m.group("place") or m.group("gplace") or m.group("wplace")
                country = SELF_PLACE.get(norm(place))
            if country and self_country is None:
                self_country = country
            self_spans.append(m.span())

    found: dict[str, list[tuple[int, int, str]]] = {}
    for a, b, _c, actors, term in hits:
        if not actors or any(s <= a < e for s, e in self_spans):
            continue
        for key in actors:
            found.setdefault(key, []).append((a, b, term))
    if not found:
        return extra, [], [], None, None, None, self_country, []

    starts = [0] + [m.end() for m in SENT_SPLIT_RE.finditer(text)]
    sent_cats: defaultdict[int, set[str]] = defaultdict(set)
    for a, _b, c, _a, _t in hits:
        if c:
            sent_cats[bisect_right(starts, a) - 1] |= c

    countries, orgs, focus = [], [], []
    terms_d, ctx_d, sup_d = {}, {}, {}
    for key, spans in sorted(found.items(), key=lambda kv: min(s[0] for s in kv[1])):
        name = ACTOR_NAME[key]
        (orgs if ACTOR_KIND[key] == "org" else countries).append(name)
        terms_d[name] = sorted({t if t.startswith("flag:") else norm(t) for _, _, t in spans})
        sent_ids: list[int] = []
        sents: list[str] = []
        near: set[str] = set()
        for a, b, _t in sorted(spans):
            i = bisect_right(starts, a) - 1
            if i in sent_ids:
                continue
            sent_ids.append(i)
            near |= sent_cats.get(i, set())
            if len(sents) < 3:
                sents.append(_sentence(text, starts, i, a, b))
        ctx_d[name] = " … ".join(sents)[:CONTEXT_MAX_CHARS]
        sup = near & SUPPORT_CATS
        if sup:
            sup_d[name] = sorted(SUPPORT_SHORT[s] for s in sup)
        if sup or ("ukraine" in near) or ({"russia", "aggression"} <= near):
            focus.append(name)
    return (extra, countries, orgs, _dump(terms_d), _dump(ctx_d), _dump(sup_d), self_country,
            focus)


@lru_cache(maxsize=None)
def country_from_flair(flair) -> str | None:
    """Author country from the flair (text or flag)."""
    if not isinstance(flair, str) or not flair.strip():
        return None
    for m in FLAG_RE.finditer(flair):
        c = ISO_TO_COUNTRY.get(flag_to_iso(m.group(0)))
        if c:
            return c
    m = FLAIR_RE.search(flair)
    if m:
        return SELF_ALL.get(norm(m.group(0)))
    return None


def is_topical(cats) -> bool:
    """Ukraine, or Russia + an aggression marker without off-topic subjects."""
    return ("ukraine" in cats) or ("russia" in cats and "aggression" in cats
                                   and "offtopic" not in cats)


# tank, missile, troops etc. are not a signal by themselves: in r/ukraine 2022 that is just
# talk about the fighting
SIGNAL_CATS: frozenset[str] = frozenset({"ukraine", "russia", "aggression"}) | SUPPORT_CATS
ANONYMOUS_AUTHORS = {"[deleted]", "[removed]", "nan", "none"}

# A comment from a topical thread/subreddit whose only link to the topic is a country name
# ("do you have indoor plumbing in France?") is about half noise and never reaches the analysis.
# True = don't keep such comments.
CONTEXT_NEEDS_TOPIC_SIGNAL = True
# officials and abbreviations missing from UKRAINE_TERMS
UKRAINE_OFFICIALS_RE = compile_terms([
    "kuleba", "reznikov", "zaluzhny", "zaluzhnyi", "yermak", "podolyak", "podoliak",
    "arestovych", "arestovich", "shmyhal", "budanov", "danilov", "klitschko", "klychko",
    "kolomoisky", "kolomoyskyi", "kolomoyskiy", "uaf",
])


def has_own_signal(text: str, cats, hits) -> bool:
    """
    Whether the text has its own topic signal: Ukraine, Russia, aggression or a support type,
    or a country together with a Ukrainian official or a description of support.
    """
    if cats & SIGNAL_CATS:
        return True
    has_actor = (any(actors for _a, _b, _c, actors, _t in hits)
                 or bool(ACTOR_SPECIAL_RE.search(text)))
    if not has_actor or not CONTEXT_NEEDS_TOPIC_SIGNAL:
        return has_actor
    return bool(UKRAINE_OFFICIALS_RE.search(text)) or any(
        m.lastgroup in SIGNAL_CATS for m in CAT_SPECIAL_RE.finditer(text))


# ============================================================
# 5. HELPERS: READING, TIME, PERIODS
# ============================================================

INVASION = pd.Timestamp(INVASION_DATE, tz="UTC")
WINDOWS = [(pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC") + pd.Timedelta(days=1))
           for s, e in PERIOD_WINDOWS]


def read_header(path: Path) -> list[str]:
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        header = fh.readline()
    return next(csv.reader([header]), [])


def read_csv_chunks(path: Path, usecols: list[str]) -> Any:
    """CSV in chunks, everything as strings, no NaN (pandas >= 1.3)."""
    return pd.read_csv(str(path), chunksize=CHUNK_SIZE, usecols=usecols, dtype=str,
                       na_filter=False, encoding="utf-8", encoding_errors="replace",
                       on_bad_lines="skip")


def fmt_time(seconds: float) -> str:
    """4156 -> "01:09:16"."""
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class FileProgress:
    """Progress by file bytes; shows nothing without tqdm."""

    def __init__(self, path: Path, desc: str):
        self.total = path.stat().st_size
        self.bar = (tqdm(total=self.total, unit="B", unit_scale=True, unit_divisor=1024,
                         desc=desc, dynamic_ncols=True) if tqdm is not None else None)

    def update_to(self, position: int):
        if self.bar is not None:
            self.bar.update(max(0, min(position, self.total) - self.bar.n))

    def postfix(self, **values):
        if self.bar is not None:
            self.bar.set_postfix(values, refresh=False)

    def close(self):
        if self.bar is not None:
            self.update_to(self.total)
            self.bar.close()


def iter_csv_chunks(path: Path, usecols: list[str], progress: FileProgress):
    """read_csv_chunks with progress updates."""
    with open(path, "rb") as fh:
        reader = pd.read_csv(fh, chunksize=CHUNK_SIZE, usecols=usecols, dtype=str,
                             na_filter=False, encoding="utf-8", encoding_errors="replace",
                             on_bad_lines="skip")
        for chunk in reader:
            yield chunk
            progress.update_to(fh.tell())


def file_kind(path: Path) -> str | None:
    n = path.name.lower()
    if "submission" in n or n.startswith("rs_"):
        return "submission"
    if "comment" in n or n.startswith("rc_"):
        return "comment"
    return None


def str_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column as strings, or empty strings if it is missing."""
    if name in df.columns:
        return pd.Series(df[name], index=df.index).astype(str)
    return pd.Series("", index=df.index, dtype=object)


def get_raw_text(chunk: pd.DataFrame) -> pd.Series:
    if "text" in chunk.columns:
        return str_col(chunk, "text")
    parts = [str_col(chunk, c) for c in ("title", "selftext", "body") if c in chunk.columns]
    if not parts:
        raise ValueError("No text column found (text / body / title / selftext)")
    s = parts[0]
    for p in parts[1:]:
        s = s + " " + p
    return s.str.strip()


def parse_created(chunk: pd.DataFrame) -> pd.Series:
    if "datetime" in chunk.columns:
        created = pd.to_datetime(chunk["datetime"], format="%Y-%m-%d %H:%M:%S",
                                 errors="coerce", utc=True)
        if created.isna().mean() > 0.5:
            created = pd.to_datetime(chunk["datetime"], errors="coerce", utc=True)
        return created
    if "created_utc" in chunk.columns:
        secs = to_num(str_col(chunk, "created_utc"))
        return pd.to_datetime(secs, unit="s", utc=True)
    raise ValueError("No date column found (datetime / created_utc)")


def window_mask(created: pd.Series) -> pd.Series:
    mask = pd.Series(False, index=created.index)
    for start, end in WINDOWS:
        mask |= (created >= start) & (created < end)
    return mask


def assign_period(created: pd.Series) -> pd.Series:
    year = created.dt.year.astype(str)
    war = created >= INVASION
    period = year.copy()
    period[(created.dt.year == INVASION.year) & ~war] = f"{INVASION.year}_pre"
    period[war] = year[war] + "_war"
    return period


# Not stored in the file since they follow from other columns; load_dataset() adds them
DATE_PARTS = ["week", "month", "period", "days_since_invasion"]     # from date
DERIVED_COLUMNS = [*DATE_PARTS, "mentions_ukraine"]                  # + from topic_source


def add_date_parts(df: pd.DataFrame, parts: list[str] | None = None,
                   as_text: bool = False) -> pd.DataFrame:
    """
    Adds week (Monday), month, period (2019 / 2022_pre / 2022_war) and days_since_invasion
    (empty for 2019) from date. parts = which ones, as_text = week as a string.
    """
    parts = DATE_PARTS if parts is None else parts
    d = pd.Series(pd.to_datetime(df["date"], errors="coerce"), index=df.index)
    d_utc = d.dt.tz_localize("UTC")          # dates are UTC, like INVASION
    if "week" in parts:
        week = d - pd.to_timedelta(d.dt.dayofweek, unit="D")
        df["week"] = week.dt.strftime("%Y-%m-%d") if as_text else week
    if "month" in parts:
        df["month"] = d.dt.strftime("%Y-%m")
    if "period" in parts:
        df["period"] = assign_period(d_utc)
    if "days_since_invasion" in parts:
        days = (d_utc - INVASION).dt.days
        df["days_since_invasion"] = days.where(d.dt.year >= INVASION.year).astype("Int32")
    return df


def subreddit_topical(subreddit: pd.Series, period: pd.Series) -> pd.Series:
    mask = pd.Series(False, index=subreddit.index)
    for sub, periods in DEDICATED_SUBREDDITS.items():
        m = subreddit.eq(sub)
        mask |= m if periods is None else (m & period.isin(periods))
    return mask


def bot_author_mask(author: pd.Series) -> pd.Series:
    low = author.str.lower()
    return low.isin(BOT_AUTHORS) | low.str.contains(r"(?:bot|b0t)$", regex=True)


def to_num(s: pd.Series) -> pd.Series:
    """pd.to_numeric with NaN instead of errors, always a Series."""
    return pd.Series(pd.to_numeric(s, errors="coerce"), index=s.index)


def none_if_empty(s: pd.Series) -> pd.Series:
    s = s.astype(object)
    return s.where(s.notna() & (s != ""), None)


# ============================================================
# 6. OUTPUT SCHEMA
# ============================================================

FLAG_COLUMNS = list(SUPPORT_TERMS.keys())

# columns of the final dataset
CORE_COLUMNS = [
    "id", "type", "subreddit",
    "date",
    "author_country", "score",
    "thread_id",
    "text", "topic_source",
    "us_domestic", "is_lifestyle",      # excluded from the analysis
    "countries", "organizations", "countries_focus",
    *FLAG_COLUMNS,                      # support types
    "country_support",
]
# for checking the filter, written only with SAVE_DEBUG_COLUMNS = True
DEBUG_COLUMNS = [
    "author", "thread_title",
    "created_at", "parent_id", "n_words", "lang", "n_countries", "author_country_source",
    "num_comments", "upvote_ratio", "mentions_russia",
    "permalink", "country_terms", "country_context",
]
PART_COLUMNS = CORE_COLUMNS + (DEBUG_COLUMNS if SAVE_DEBUG_COLUMNS else [])
FINAL_COLUMNS = PART_COLUMNS + ["thread_n", "subreddit_day_total", "subreddit_month_total"]
LIST_COLUMNS = ["countries", "organizations", "countries_focus"]
# added by add_reactions.py
REACTION_COLUMNS = ["reaction_ukraine", "reaction_russia", "reaction"]
BOOL_COLUMNS = ["mentions_ukraine", "mentions_russia", "us_domestic", "is_lifestyle",
                *FLAG_COLUMNS]


# ============================================================
# 7. STEP 1: THREAD INDEX (title + whether the thread is on topic)
# ============================================================

THREAD_CATS = frozenset(["us_domestic", "lifestyle", "war", "ukraine"])


def build_thread_index(files: list[Path]) -> dict:
    """Post id -> (title, on topic or not, categories for flags)."""
    threads: dict[str, tuple] = {}
    for idx, path in enumerate(files, 1):
        cols = read_header(path)
        usecols = [c for c in ("id", "type", "title", "text", "selftext", "body") if c in cols]
        if "id" not in usecols:
            continue
        progress = FileProgress(path, f"[{idx}/{len(files)}] {path.name}")
        if tqdm is None:
            print(f"  {path.name}")
        for chunk in iter_csv_chunks(path, usecols, progress):
            if "type" in chunk.columns:
                chunk = chunk[chunk["type"].str.strip().str.lower().eq("submission")]
            if chunk.empty:
                continue
            clean = get_raw_text(chunk).map(clean_text).tolist()
            title = (chunk["title"].map(clean_text).tolist() if "title" in chunk.columns
                     else clean)
            for sid, ttl, txt in zip(chunk["id"].str.strip().tolist(), title, clean):
                cats = full_categories(txt)
                threads[sid] = (ttl[:THREAD_TITLE_MAX_CHARS], is_topical(cats),
                                frozenset(cats & THREAD_CATS))
            progress.postfix(posts=f"{len(threads):,}")
        progress.close()
    return threads


# ============================================================
# 8. STEP 2: PROCESSING CHUNKS
# ============================================================

class State:
    def __init__(self):
        self.den_month = Counter()          # (subreddit, YYYY-MM, type) -> valid count
        self.den_day = Counter()            # (subreddit, YYYY-MM-DD, type) -> valid count
        self.den_period = Counter()         # period -> valid count
        self.rows_read = 0
        self.rows_valid = 0
        self.rows_kept = 0
        self.part_no = 0
        self.seen: set = set()
        self.by_period = Counter()
        self.by_subreddit = Counter()
        self.by_source = Counter()
        self.by_lang = Counter()
        self.countries_by_period = defaultdict(Counter)
        self.thread_rows = Counter()        # rows of the thread in the dataset
        self.dropped = Counter()            # drop reason -> count
        self.rng = random.Random(RANDOM_SEED)
        self.reservoirs: dict[tuple, dict] = {}

    def sample(self, label: str, frame: pd.DataFrame, by: str = "period",
               k: int = VALIDATION_PER_PERIOD):
        """Reservoir sampling, k rows per value of `by`."""
        for key, grp in frame.groupby(by, sort=False):
            r = self.reservoirs.setdefault((label, key), {"seen": 0, "items": []})
            for pos in range(len(grp)):
                r["seen"] += 1
                if len(r["items"]) < k:
                    r["items"].append(grp.iloc[pos].to_dict())
                else:
                    j = self.rng.randrange(r["seen"])
                    if j < k:
                        r["items"][j] = grp.iloc[pos].to_dict()


def process_chunk(chunk: pd.DataFrame, default_type: str | None, threads: dict,
                  state: State) -> pd.DataFrame | None:
    state.rows_read += len(chunk)

    # type, time, author; no bots, nothing outside the periods
    if "type" in chunk.columns:
        rtype = chunk["type"].str.strip().str.lower()
        rtype = rtype.where(rtype.isin(["submission", "comment"]), default_type or "comment")
    else:
        rtype = pd.Series(default_type or "comment", index=chunk.index)
    created = parse_created(chunk)
    author = (chunk["author"].astype(str).str.strip() if "author" in chunk.columns
              else pd.Series("", index=chunk.index))
    valid = window_mask(created) & ~bot_author_mask(author)
    if not valid.any():
        return None
    chunk = chunk.loc[valid]

    def col(name: str) -> pd.Series:
        return str_col(chunk, name)

    df = pd.DataFrame({
        "id": col("id").str.strip(),
        "type": rtype[valid],
        "subreddit": col("subreddit").str.strip().str.lower(),
        "created_at": created[valid],
        "author": author[valid],
        "raw": get_raw_text(chunk),
    })
    df["month"] = df["created_at"].dt.strftime("%Y-%m")
    day = df["created_at"].dt.strftime("%Y-%m-%d")
    df["period"] = assign_period(pd.Series(df["created_at"], index=df.index))

    # denominators: all valid texts of the subreddit per day / month / period
    state.rows_valid += len(df)
    state.den_month.update(Counter(zip(df["subreddit"], df["month"], df["type"])))
    state.den_day.update(Counter(zip(df["subreddit"], day, df["type"])))
    state.den_period.update(df["period"].tolist())

    # thread and subreddit
    is_comment = df["type"].eq("comment")
    link = col("link_id").str.strip().str.replace(r"^t3_", "", regex=True)
    thread_key = link.where(is_comment, df["id"])
    info = thread_key.map(threads)
    has_info = info.map(lambda x: isinstance(x, tuple))
    thread_title = info.map(lambda x: x[0] if isinstance(x, tuple) else "")
    thread_topical = (info.map(lambda x: x[1] if isinstance(x, tuple) else False)
                      .astype(bool) & is_comment & has_info)
    thread_cats = info.map(lambda x: x[2] if isinstance(x, tuple) else EMPTY)
    sub_topical = subreddit_topical(str_col(df, "subreddit"), str_col(df, "period"))

    # candidates: quick check of the raw text so we don't clean everything
    cand = (thread_topical | sub_topical).to_numpy().copy()
    rest = ~cand
    if rest.any():
        cand[rest] = df.loc[rest, "raw"].str.contains(UKR_RUS_RE).to_numpy()
    cidx = df.index[cand]

    kidx = pd.Index([])
    kept_idx = pd.Index([])
    # placeholders so the names exist when there are no candidates
    text = n_words = lang = source = ukr = rus = hits = cats = pd.Series(dtype=object)
    if len(cidx):
        text = df.loc[cidx, "raw"].map(clean_text)
        tlist = text.tolist()
        stats = [text_stats(t) for t in tlist]
        scans = [scan(t) for t in tlist]
        n_words = pd.Series([s[0] for s in stats], index=cidx)
        lang = pd.Series([s[1] for s in stats], index=cidx)
        cats = pd.Series([s[0] for s in scans], index=cidx)
        hits = pd.Series([s[1] for s in scans], index=cidx)
        is_short = n_words < MIN_WORDS
        is_bot = text.str.contains(BOT_TEXT_RE) | text.str.contains(MOD_TEXT_RE)
        state.dropped["short text"] += int(is_short.sum())
        state.dropped["bot / moderator"] += int((is_bot & ~is_short).sum())
        ok = (~is_short & ~is_bot).to_numpy()
        oidx = cidx[ok]

        oc = cats.loc[oidx]
        ukr = oc.map(lambda c: "ukraine" in c).astype(bool)
        rus = oc.map(lambda c: "russia" in c).astype(bool)
        rus_topic = oc.map(lambda c: is_topical(c) and "ukraine" not in c).astype(bool)
        # a comment that got in only via its thread/subreddit needs its own signal
        signal = pd.Series([has_own_signal(t, c, h) for t, c, h in
                            zip(text.loc[oidx].tolist(), oc.tolist(),
                                hits.loc[oidx].tolist())], index=oidx, dtype=bool)
        thread_ok = (thread_topical.loc[oidx] & signal).to_numpy()
        sub_ok = (sub_topical.loc[oidx] & signal).to_numpy()
        source = pd.Series(np.select(
            [ukr.to_numpy(), rus_topic.to_numpy(), thread_ok, sub_ok],
            ["text_ukraine", "text_russia", "thread", "subreddit"], default=""),
            index=oidx)
        kidx = oidx[source.ne("").to_numpy()]
        via_context = ((thread_topical.loc[oidx] | sub_topical.loc[oidx])
                       & ~ukr & ~rus_topic)
        state.dropped["thread/subreddit, no own signal"] += int((via_context & ~signal).sum())

        # duplicates within the file: same id, or same text from the same author
        fresh = []
        for rid, auth, txt in zip((df.loc[kidx, "type"].str[0] + df.loc[kidx, "id"]).tolist(),
                                  df.loc[kidx, "author"].tolist(), text.loc[kidx].tolist()):
            keys = [rid]
            if auth and auth.lower() not in ANONYMOUS_AUTHORS:
                keys.append(hash((auth.lower(), " ".join(txt.lower().split()))))
            is_new = not any(k in state.seen for k in keys)
            fresh.append(is_new)
            state.seen.update(keys)
        state.dropped["duplicate / copy-paste"] += len(fresh) - sum(fresh)
        kept_idx = kidx[np.array(fresh, dtype=bool)] if fresh else kidx

    # sample of dropped rows, to check we don't lose relevant ones
    if SAVE_VALIDATION_SAMPLE:
        dropped = df.index.difference(kidx)
        if len(dropped):
            fr = df.loc[dropped, ["id", "type", "subreddit", "period", "raw"]].copy()
            fr["permalink"] = col("permalink").loc[dropped]
            state.sample("dropped", fr)

    if len(kept_idx) == 0:
        return None

    # features for kept texts
    out = df.loc[kept_idx, ["id", "type", "subreddit", "created_at", "author",
                            "month", "period"]].copy()
    kt = text.loc[kept_idx].tolist()
    out["text"] = kt
    out["n_words"] = n_words.loc[kept_idx].astype("int32")
    out["lang"] = lang.loc[kept_idx]
    out["topic_source"] = source.loc[kept_idx]
    out["mentions_ukraine"] = ukr.loc[kept_idx]
    out["mentions_russia"] = rus.loc[kept_idx]

    det = [analyze_details(t, h) for t, h in zip(kt, hits.loc[kept_idx].tolist())]
    all_cats = [c | d[0] for c, d in zip(cats.loc[kept_idx].tolist(), det)]
    tcats = thread_cats.loc[kept_idx].tolist()

    for name in FLAG_COLUMNS:
        out[name] = [name in c for c in all_cats]
    any_support = out[FLAG_COLUMNS].any(axis=1).to_numpy()
    # for comments the thread post counts too
    titles = thread_title.loc[kept_idx].fillna("").astype(str).tolist()
    scandal = out["period"].isin(US_SCANDAL_PERIODS).tolist()
    out["us_domestic"] = [
        ("us_domestic" in c) or ("us_domestic" in t)
        # 2019: Trump + Ukraine = impeachment
        or (sc and bool(TRUMP_RE.search(txt) or TRUMP_RE.search(ttl))
            and (("ukraine" in c) or ("ukraine" in t)))
        for c, t, sc, txt, ttl in zip(all_cats, tcats, scandal, kt, titles)]
    life = np.array([(("lifestyle" in c) or ("lifestyle" in t)) and
                     not (("war" in c) or ("war" in t)) for c, t in zip(all_cats, tcats)])
    out["is_lifestyle"] = life & ~any_support

    out["countries"] = [d[1] for d in det]
    out["organizations"] = [d[2] for d in det]
    out["n_countries"] = np.array([len(d[1]) for d in det], dtype="int16")
    out["country_terms"] = [d[3] for d in det]
    out["country_context"] = [d[4] for d in det]
    out["country_support"] = [d[5] for d in det]
    out["countries_focus"] = [d[7] for d in det]
    out["thread_id"] = thread_key.loc[kept_idx]

    # author country: flair, otherwise self-identification in the text
    flair_c = col("author_flair").loc[kept_idx].map(country_from_flair)
    self_c = pd.Series([d[6] for d in det], index=kept_idx, dtype=object)
    has_flair = flair_c.notna()
    out["author_country"] = flair_c.where(has_flair, self_c)
    source_country = pd.Series(None, index=kept_idx, dtype=object)
    source_country[self_c.notna()] = "text"
    source_country[has_flair] = "flair"
    out["author_country_source"] = source_country

    # week, period etc. are derived from date by load_dataset
    out["date"] = out["created_at"].dt.date

    # engagement and links
    is_sub = out["type"].eq("submission")
    out["score"] = (to_num(col("score").loc[kept_idx])
                    .round().astype("Int64"))
    nc = to_num(col("num_comments").loc[kept_idx]).round()
    out["num_comments"] = nc.where(is_sub).astype("Int64")
    ur = to_num(col("upvote_ratio").loc[kept_idx])
    out["upvote_ratio"] = ur.where(is_sub).astype("float32")
    out["link_id"] = none_if_empty(col("link_id").loc[kept_idx])
    out["parent_id"] = none_if_empty(col("parent_id").loc[kept_idx])
    out["permalink"] = none_if_empty(col("permalink").loc[kept_idx])
    out["thread_title"] = none_if_empty(thread_title.loc[kept_idx])

    # stats
    state.rows_kept += len(out)
    state.by_period.update(out["period"].tolist())
    state.by_subreddit.update(out["subreddit"].tolist())
    state.by_source.update(out["topic_source"].tolist())
    state.by_lang.update(out["lang"].tolist())
    state.thread_rows.update(out["thread_id"].tolist())
    for p, lst in zip(out["period"], out["countries"]):
        if lst:
            state.countries_by_period[p].update(lst)

    if SAVE_VALIDATION_SAMPLE:
        fr = out[["id", "type", "subreddit", "period", "permalink", "topic_source", "text"]].copy()
        fr["countries"] = ["|".join(x) for x in out["countries"]]
        fr["organizations"] = ["|".join(x) for x in out["organizations"]]
        state.sample("kept", fr)

    if SAVE_PREVIEW:
        pv = out[["id", "type", "subreddit", "month", "date", "topic_source", "us_domestic",
                  "is_lifestyle", "thread_title", "text", "permalink"]].copy()
        pv["date"] = pv["date"].astype(str)
        for c in LIST_COLUMNS:
            pv[c] = ["|".join(x) for x in out[c]]
        pv["support"] = [",".join(SUPPORT_SHORT[f] for f, v in zip(FLAG_COLUMNS, vals) if v)
                         for vals in out[FLAG_COLUMNS].itertuples(index=False)]
        state.sample("preview", pv, by="month", k=PREVIEW_PER_MONTH)

    return out


def write_part(out: pd.DataFrame, state: State):
    """Temporary CSV part, lists written as "Poland|Germany"."""
    part = out.copy()
    if "created_at" in PART_COLUMNS:
        part["created_at"] = part["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in LIST_COLUMNS:
        part[c] = ["|".join(x) for x in part[c]]
    for c in BOOL_COLUMNS:                       # 1/0 instead of True/False
        part[c] = part[c].astype(bool).astype(int)
    part[PART_COLUMNS].to_csv(TMP_DIR / f"part_{state.part_no:05d}.csv", index=False,
                              encoding="utf-8")
    state.part_no += 1


def process_file(path: Path, threads: dict, state: State, desc: str = ""):
    cols = read_header(path)
    usecols = [c for c in WANTED_COLUMNS if c in cols]
    kind = file_kind(path)
    if "link_id" not in cols and kind != "submission":
        print("  warning: no link_id, comments will not inherit the thread topic")
    state.seen = set()   # duplicates within one file (month)
    t_file = time.time()
    read_before, kept_before = state.rows_read, state.rows_kept
    progress = FileProgress(path, desc or path.name)
    for i, chunk in enumerate(iter_csv_chunks(path, usecols, progress), 1):
        before = state.rows_kept
        t0 = time.time()
        out = process_chunk(chunk, kind, threads, state)
        if out is not None and len(out):
            write_part(out, state)
        progress.postfix(read=f"{state.rows_read - read_before:,}",
                         kept=f"{state.rows_kept - kept_before:,}")
        if tqdm is None:
            print(f"  chunk {i}: read {len(chunk):,}, kept "
                  f"{state.rows_kept - before:,}  ({fmt_time(time.time() - t0)})")
    progress.close()
    n_read, n_kept = state.rows_read - read_before, state.rows_kept - kept_before
    share = n_kept / n_read if n_read else 0
    print(f"  read {n_read:,} | kept {n_kept:,} ({share:.1%}) | "
          f"{fmt_time(time.time() - t_file)}")


# ============================================================
# 9. STEP 3: ONE FILE + DENOMINATORS
# ============================================================

def _alt_name(target: Path) -> Path:
    return target.with_name(f"{target.stem}_{time.strftime('%Y%m%d_%H%M%S')}{target.suffix}")


def safe_replace(tmp: Path, target: Path, attempts: int = 30, wait: int = 10) -> Path:
    """
    os.replace with retries: on Windows the target can't be overwritten while it is open
    in another program. If it stays locked, saves under another name. Returns the path.
    """
    for i in range(attempts):
        try:
            os.replace(tmp, target)
            return target
        except PermissionError:
            if i == 0:
                print(f"\nCannot overwrite {target.name}: it is open in another program "
                      f"(PyCharm, Excel...).\n  Close it, retrying every {wait} s "
                      f"(up to {attempts * wait // 60} min).")
            time.sleep(wait)
    alt = _alt_name(target)
    os.replace(tmp, alt)
    print(f"{target.name} is still locked, saved as {alt.name} instead")
    return alt


def safe_to_csv(df: Any, target: Path, **kwargs) -> Path:
    """to_csv that saves under another name if the file is locked."""
    try:
        df.to_csv(target, **kwargs)
        return target
    except PermissionError:
        alt = _alt_name(target)
        df.to_csv(alt, **kwargs)
        print(f"{target.name} is open in another program, saved as {alt.name}")
        return alt


def merge_parts(state: State) -> Path:
    """Merges the parts into one CSV and adds denominators (via .tmp)."""
    parts = sorted(TMP_DIR.glob("part_*.csv"))
    tmp = OUTPUT_CSV.with_name(OUTPUT_CSV.name + ".tmp")
    header = True
    bar = (tqdm(parts, desc="Merging", unit=" parts", dynamic_ncols=True)
           if tqdm is not None else parts)
    for p in bar:
        d = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")
        d["thread_n"] = [state.thread_rows.get(t, 0) for t in d["thread_id"]]
        d["subreddit_day_total"] = [state.den_day.get(k, 0) for k in
                                    zip(d["subreddit"], d["date"], d["type"])]
        d["subreddit_month_total"] = [state.den_month.get(k, 0) for k in
                                      zip(d["subreddit"], d["date"].str[:7], d["type"])]
        d[FINAL_COLUMNS].to_csv(tmp, mode="w" if header else "a", header=header,
                                index=False, encoding="utf-8")
        header = False
    return safe_replace(tmp, OUTPUT_CSV)


def load_dataset(path: Path = OUTPUT_CSV, usecols: list[str] | None = None,
                 nrows: int | None = None) -> pd.DataFrame:
    """
    Reads the dataset with types: country lists as list, date/week as dates, flags as bool.
    Adds week, month, period, days_since_invasion and mentions_ukraine; they can be
    requested in usecols like normal columns.
    """
    want = None if usecols is None else list(usecols)
    read_cols = None
    if want is not None:
        read_cols = [c for c in want if c not in DERIVED_COLUMNS]
        if any(c in DATE_PARTS for c in want) and "date" not in read_cols:
            read_cols.append("date")
        if "mentions_ukraine" in want and "topic_source" not in read_cols:
            read_cols.append("topic_source")
    df = pd.read_csv(path, usecols=read_cols, nrows=nrows, keep_default_na=False,
                     na_values=[""], low_memory=False, encoding="utf-8")
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        add_date_parts(df, DATE_PARTS if want is None else [c for c in DATE_PARTS if c in want])
    if "topic_source" in df.columns and (want is None or "mentions_ukraine" in want):
        df["mentions_ukraine"] = df["topic_source"].eq("text_ukraine")
    for c in LIST_COLUMNS:
        if c in df.columns:
            df[c] = df[c].fillna("").map(lambda x: x.split("|") if x else [])
    for c in BOOL_COLUMNS:
        if c in df.columns and df[c].dtype != bool:
            df[c] = df[c].astype(str).isin(["1", "1.0", "True", "true"])
    if want is not None:
        return df.loc[:, want]
    # derived columns go right after the ones they come from
    order = [c for c in df.columns if c not in DERIVED_COLUMNS]
    for anchor, extra in (("date", DATE_PARTS), ("topic_source", ["mentions_ukraine"])):
        if anchor in order:
            pos = order.index(anchor) + 1
            order[pos:pos] = [c for c in extra if c in df.columns]
    return df.loc[:, order]


def save_validation(state: State):
    rows = []
    for (label, period), r in sorted(state.reservoirs.items()):
        if label not in ("kept", "dropped"):
            continue
        for item in r["items"]:
            item = dict(item)
            if "raw" in item:
                item["text"] = clean_text(item.pop("raw"))
            item["sample"] = label
            rows.append(item)
    if not rows:
        return
    v = pd.DataFrame(rows)
    for c in ("topic_source", "countries", "organizations"):
        if c not in v.columns:
            v[c] = ""
    v["is_relevant (1/0)"] = ""
    v["countries_correct (1/0)"] = ""
    v["note"] = ""
    v = v[["sample", "period", "subreddit", "type", "id", "permalink", "topic_source",
           "countries", "organizations", "text",
           "is_relevant (1/0)", "countries_correct (1/0)", "note"]]
    safe_to_csv(v, VALIDATION_FILE, index=False, encoding="utf-8-sig")


def save_preview(state: State):
    """PREVIEW_PER_MONTH random rows per month to look through."""
    rows = [item for (label, _m), r in state.reservoirs.items() if label == "preview"
            for item in r["items"]]
    if not rows:
        return
    v = pd.DataFrame(rows).sort_values(["month", "date", "subreddit"])
    v["us_domestic"] = v["us_domestic"].astype(bool).astype(int)
    v["is_lifestyle"] = v["is_lifestyle"].astype(bool).astype(int)
    cols = ["month", "date", "subreddit", "type", "topic_source", "countries",
            "countries_focus", "organizations", "support", "us_domestic", "is_lifestyle",
            "thread_title", "text", "permalink", "id"]
    safe_to_csv(v[cols], PREVIEW_FILE, index=False, encoding="utf-8-sig")


def coverage_gaps(state: State) -> list[str]:
    """Months with no posts or comments for a subreddit (usually incomplete parser files)."""
    months = sorted({m for (_s, m, _t) in state.den_month})
    subs = sorted({sub for (sub, _m, _t) in state.den_month})
    gaps = []
    for sub in subs:
        for typ, label in (("submission", "submissions"), ("comment", "comments")):
            have = [m for m in months if state.den_month.get((sub, m, typ), 0) > 0]
            if not have:
                continue
            # empty months are normal for small subreddits
            sizes = sorted(state.den_month[(sub, m, typ)] for m in have)
            if sizes[len(sizes) // 2] < GAP_MIN_MONTHLY:
                continue
            missing = [m for m in months if m not in have]
            if missing:
                gaps.append(f"r/{sub}: no {label} for {', '.join(missing)}")
    return gaps


# ============================================================
# 10. MAIN
# ============================================================

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(f"No {INPUT_GLOB} files in {INPUT_DIR}")

    print("=" * 70)
    print("REDDIT -> UKRAINE SUPPORT DATASET")
    print("=" * 70)
    print(f"Input folder: {INPUT_DIR}")
    print(f"Files:        {len(files)}")
    print(f"Periods:      {', '.join(f'{a} to {b}' for a, b in PERIOD_WINDOWS)}")

    print("\nStep 1/3. Thread index (post titles)")
    sub_files = [f for f in files if file_kind(f) in ("submission", None)]
    threads = build_thread_index(sub_files)
    n_top = sum(1 for v in threads.values() if v[1])
    print(f"  threads: {len(threads):,}, topical: {n_top:,}")

    print(f"  time: {fmt_time(time.time() - t0)}")

    print("\nStep 2/3. Filtering")
    t_step = time.time()
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)
    state = State()
    for idx, path in enumerate(files, 1):
        if tqdm is None:
            print(f"\n[{idx}/{len(files)}] {path.name}")
        process_file(path, threads, state, desc=f"[{idx}/{len(files)}] {path.name}")
    print(f"  filtering time: {fmt_time(time.time() - t_step)}  "
          f"(total: {fmt_time(time.time() - t0)})")

    if state.part_no == 0:
        print("\nNo topical records found.")
        return

    print("\nStep 3/3. Merging into one file and adding denominators")
    t_step = time.time()
    final_csv = merge_parts(state)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    if SAVE_VALIDATION_SAMPLE:
        save_validation(state)
    if SAVE_PREVIEW:
        save_preview(state)
    print(f"  time: {fmt_time(time.time() - t_step)}")

    # summary
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Rows read:                   {state.rows_read:,}")
    print(f"Valid (in period, no bots):  {state.rows_valid:,}")
    print(f"In dataset:                  {state.rows_kept:,}")
    if state.dropped:
        print("\nDropped among candidates:")
        for reason, n in state.dropped.most_common():
            print(f"  {reason:<38} {n:>10,}")

    print("\nBy period (in dataset / all valid / share):")
    for p in sorted(state.den_period):
        kept, total = state.by_period.get(p, 0), state.den_period[p]
        print(f"  {p:<10} {kept:>10,} / {total:>12,}   {kept / total:6.1%}")
    print("\nTopic source:")
    for s, n in state.by_source.most_common():
        print(f"  {s:<14} {n:>10,}")
    print("\nSubreddits:")
    for s, n in state.by_subreddit.most_common():
        print(f"  r/{s:<22} {n:>10,}")
    print("\nLanguage:")
    for s, n in state.by_lang.most_common():
        print(f"  {s:<10} {n:>10,}")
    for p in sorted(state.countries_by_period):
        print(f"\nTop 15 countries, {p}:")
        for c, n in state.countries_by_period[p].most_common(15):
            print(f"  {c:<15} {n:>10,}")

    gaps = coverage_gaps(state)
    if gaps:
        print("\nWarning: gaps in the input data (check the parser output).")
        print("  Without submissions (RS_YYYY-MM.zst) there are no posts in the dataset,")
        print("  and comments cannot inherit the thread topic:")
        for g in gaps:
            print("  " + g)

    size_mb = final_csv.stat().st_size / 1024 ** 2
    print(f"\nDataset: {final_csv}  ({size_mb:,.1f} MB)")
    if SAVE_VALIDATION_SAMPLE:
        print(f"Validation sample: {VALIDATION_FILE}")
    if SAVE_PREVIEW:
        print(f"Preview ({PREVIEW_PER_MONTH} rows per month): {PREVIEW_FILE}")
    print(f"Total time: {fmt_time(time.time() - t0)} (h:m:s)")


if __name__ == "__main__":
    main()