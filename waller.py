# -*- coding: utf-8 -*-
"""
@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

import requests
import json
import os.path
import re

if __name__ == '__main__':
    from base import DataLoaderBase
else:
    from .base import DataLoaderBase


class Waller(DataLoaderBase):
    def __init__(self, datapath, verbose=False):
        super(Waller, self).__init__()
        self.reprname = "Waller"
        assert os.path.exists(datapath), "Path not found"
        self.datapath = datapath
        self.verbose = verbose
        filename = os.path.join(self.datapath, "metadata.json.gz")
        if os.path.exists(filename):
            import gzip
            f = gzip.open(filename, 'r')
            self.data = json.loads(f.read().decode('utf-8'))
            f.close()
        else:
            self.data = {}
    
    def save(self):
        filename = os.path.join(self.datapath, "metadata.json.gz")
        import gzip
        f = gzip.open(filename, 'w')
        f.write(json.dumps(self.data, sort_keys=True, indent=2, separators=(',', ': ')).encode('utf-8'))
        f.close()

    def getNumbers(self):
        return [int(key) for key in self.data.keys() if self.data[key]['response_status_code'] == 200]

    def keys(self):
        return self.getNumbers()

    def dataIterator(self, numbers):
        import copy
        for number in numbers:
            data = copy.deepcopy(self[number])
            data['id'] = number
            yield data

    def populate(self):
        downloadCounter = 0
        for number in range(10000, 50000):
            if str(number) in self.data.keys():
                respose_code = self.data[str(number)]['response_status_code']
                if self.verbose:
                    print("%i, already checked, response was %i" % (number, respose_code))
            else:
                respose_code = 0
            if not (respose_code == 200 or respose_code == 404):
                self._download(number)
                downloadCounter += 1
                if downloadCounter > 1000:
                    if self.verbose:
                        print("Writing database to file")
                    self.save()
                    downloadCounter = 0
        self.save()

    def __getitem__(self, number):
        import copy
        return copy.deepcopy(self.data[str(number)])

    def _get_waller_url(self, number):
        return "http://waller.ub.uu.se/" + str(number) + ".html"

    def _getMetaDataFromTemplate(self, response, markerTemplate):
        idx = response.content.find(markerTemplate)
        if idx >= 0:
            text = response.content[idx+len(markerTemplate):]
            text = text[:text.find(markerTemplate)]
            removedSomething = True
            while removedSomething:
                removelist = ['\r', '\n', ' ']
                removedSomething = False
                for t in removelist:
                    if text[0] == t:
                        removedSomething = True
                        text = text[1:]
                if text[0] == '<':
                    removedSomething = True
                    while text[0] != '>':
                        text = text[1:]
                    text = text[1:]
            return text[:text.find('<')]
        else:
            return None

    def _trimUntil(self, text, template):
        idx = text.find(template)
        if idx >= 0:
            return text[idx+len(template):]
        else:
            return text
    
    def _trimFrom(self, text, template):
        idx = text.find(template)
        if idx >= 0:
            return text[:idx]
        else:
            return text
    
    def _ltrim(self, text):
        while len(text)>0 and text[0] == ' ':
            text = text[1:]
        return text
    
    def _rtrim(self, text):
        while len(text)>0 and text[-1] == ' ':
            text = text[:-1]
        return text

    def _tagRemover(self, text):
        s = ""
        l = []
        w = True
        for c in text:
            if c == '<':
                w = False
            if w and c != '\n':
                s += c
            if c == '>':
                w = True
                s = self._rtrim(self._ltrim(s))
                if len(s) > 0:
                    l.append(s)
                s = ""
        return l

    def _download(self, number):
        if self.verbose:
            print ("%i, checking..." % number, end="")
        response = requests.get(self._get_waller_url(number))
        if self.verbose:
            print("response was %i" % response.status_code)
        newentry = {'response_status_code': response.status_code}
        if response.status_code == 200:
            response_content = response.content.decode('utf-8')
            text = self._trimUntil(response_content, "PHYSICAL DESCRIPTION")
            text = self._trimUntil(text, "</tr>")
            text = self._trimFrom(text, "DESCRIPTION OF CONTENTS")
            entries1 = self._tagRemover(text)
        
            tags = [("Shelfmark:", "shelfmark"), ("Type of object:", "type_of_object"), 
                    ("Dimensions:", "dimensions"), ("Extent:", "extent"), ("Material:", "material")]
            def parseSingleTags(entries, tags):
                physical = {}
                i = 0        
                while i < len(entries):
                    for texttag, dictkey in tags:
                        if entries[i].find(texttag) >= 0:
                            i += 1
                            physical[dictkey] = entries[i]
                    i += 1
                return physical
            newentry['physical_description'] = parseSingleTags(entries1, tags)
        
            text = self._trimUntil(response_content, "DESCRIPTION OF CONTENTS")
            text = self._trimUntil(text, "</tr>")
            text = self._trimFrom(text, "IMAGES")
            entries2 = self._tagRemover(text)
        
            tags = [("Type of element:", "type_of_element"), ("Extent:", "extent"), 
                    ("Language:", "language"), ("Place:", "place"), ("Date:", "date"), 
                    ("Short summary:", "short_summary")]
            content = parseSingleTags(entries2, tags)
            tags = [("Person:", "person")]
            i = 0
            while i < len(entries2):
                for texttag, dictkey in tags:
                    if entries2[i].find(texttag) >= 0:
                        i += 1
                        if dictkey in content.keys():
                            content[dictkey].append(entries2[i])
                        else:
                            content[dictkey] = [entries2[i]]
                i += 1
            newentry['description_of_contents'] = content
        
            tags = [("Comments:", "comments")]
            comment = parseSingleTags(entries2, tags)
            if 'comments' in comment.keys():
                newentry['comments'] = comment['comments']
            
            text = self._trimUntil(response_content, "IMAGES")
            images =  []
            match = re.findall(r'href=[\'"]?([^\'" >]+)', text)
            for m in match:
                u = "http://waller.ub.uu.se/images/"
                if m[:len(u)] == u:
                    images.append(m)
            newentry['image_urls'] = images
        self.data[str(number)] = newentry


if __name__=='__main__':
    datapath = "/media/fredrik/UB Storage/tmp/Waller"
    database = Waller(datapath=datapath, verbose=1)
    database.populate()
    database.save()
