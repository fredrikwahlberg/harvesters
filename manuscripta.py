# -*- coding: utf-8 -*-
import requests
import re


browse_url = 'https://www.manuscripta.se/browse/manuscripts'

def webget(url):
    response = requests.get(url)
    if response.ok:
        return response.content.decode('utf-8')
    else:
        return None


ms_ids = [int(ms) for ms in re.findall(r'(?i)<a href="\/ms\/([^>]+)">', 
                                          webget(browse_url))]

tei_urls = ["https://www.manuscripta.se/xml/" + str(n) for n in ms_ids]
from multiprocessing import Pool
import os
with Pool(processes=os.cpu_count()*2) as pool:
    tei_xml = list(pool.imap(webget, tei_urls))

t = tei_xml[160]

#"""<primaryAddress>[\s\S]*?<\/primaryAddress>"""

#re.findall(r'(?i)<surface ([^>]+)>(.+?)</surface>', t)
surfaces = [t[1] for t in re.findall(r'(?i)<surface ([^>]+?)>([\s\S]+?)</surface>', tei_xml[160])]
desc = [re.findall(r'(?i)<desc>(.+?)</desc>', s)[0] for s in surfaces]
g = [re.findall(r'(?i)<graphic ([^>]+)/>', s)[0] for s in surfaces]
o = g[0].split(' ')
#p = dict()
for p in o:
    k, v = p.split('=')
    v = v.replace('"', '')
    

def image_url(number, filename, width):
    return """https://www.manuscripta.se/iipsrv/iipsrv.fcgi?IIIF=""" + str(number) + """/""" + filename + """/full/""" + str(width) + """,/0/default.jpg"""

image_url(100379, "uub-b-023_0003_001r.tif", 3530)

#%% lang
surfaces = [t[1] for t in re.findall(r'(?i)<surface ([^>]+?)>([\s\S]+?)</surface>', tei_xml[160])]
languages = [re.findall(r'(?i)<textLang mainLang="([^>]+)"/>', text) for text in tei_xml]

#textLang mainLang=""

w1 = webget("https://www.manuscripta.se/iiif/collection-ttt.json")
import json
d = json.loads(w1)
w2 = webget("https://www.manuscripta.se/iiif/")

# TODO Switch to json and iiif

#https://www.manuscripta.se/iiif/100205/manifest.json

#https://www.manuscripta.se/iipsrv/iipsrv.fcgi?IIIF=100379/uub-b-023_0003_001r.tif/full/660,/0/default.jpg
#language
#support
#date
#shelfmark
#digitized
#images
# - url
# - width
# - height
# - 