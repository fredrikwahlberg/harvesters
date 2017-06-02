# -*- coding: utf-8 -*-
"""
@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

import requests
import json
import os.path
import zipfile
import urllib.request
#import re

def load_data_file(year):
    url = """https://data.riksdagen.se/dataset/anforande/"""
    fn = """anforande-20""" + str(int(year)) + str(int(year+1)) + """.json.zip"""
    if not os.path.exists(fn):
        print("Downloading %s" % fn)
        with urllib.request.urlopen(url+fn) as response, open(fn, 'wb') as outfile:
            data = response.read()
            outfile.write(data)
    with zipfile.ZipFile(fn, 'r') as archive:
        parsed_data = list()
        for filename in archive.filelist:
            data = archive.read(filename)
            parsed_data.append(json.loads(data.decode('utf-8-sig', 'strict')))
            #print("%s, %i bytes" % (filename, len(data)))
    # TODO Clean and reformat the data
    return parsed_data

#"""https://data.riksdagen.se/dataset/person/person.json.zip"""

if __name__ == '__main__':
    #print("""едц""")
    data = load_data_file(15)

