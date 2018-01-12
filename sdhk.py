# -*- coding: utf-8 -*-
"""
Data harvester for 'Svenskt Diplomatariums Huvudkartotek' (SDHK)

@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

import requests
import os
import os.path
import re
import numpy as np

def download_metadata(number):
    return_dict = {'sdhk' : number}
    try:
        #http://sok.riksarkivet.se/SDHK?EndastDigitaliserat=true&DatumTill=1535&Brevtext=true&Extratext=true&Sigill=true&Original=true&MedeltidaAvskrifter=false&MedeltidaRegest=false&Brocman=false&Hadorph=false&Peringskiold=false&Ornhielm=true&OvrMedeltidaAvskrifter=true&OvrMedeltidaRegest=true&TrycktUtgava=true&TrycktRegest=true&AvanceradSok=true&page=1&postid=Dipl_24744&tab=post
    #            metadataurl1 = "http://sok.riksarkivet.se/SDHK?EndastDigitaliserat=true&DatumTill=1535&Brevtext=true&Extratext=true&Sigill=true&Original=true&MedeltidaAvskrifter=false&MedeltidaRegest=false&Brocman=false&Hadorph=false&Peringskiold=false&Ornhielm=true&OvrMedeltidaAvskrifter=true&OvrMedeltidaRegest=true&TrycktUtgava=true&TrycktRegest=true&AvanceradSok=true&page=1&postid=Dipl_"
    #            metadataurl2 = "&tab=post"
        url1 = "https://sok.riksarkivet.se/sdhk?EndastDigitaliserat=false&SDHK="
        url2 = "&TrycktUtgava=true&TrycktRegest=true&Brevtext=true&Extratext=true&Sigill=true&Original=true&MedeltidaAvskrifter=true&MedeltidaRegest=true&EftermedeltidaAvskrifter=true&EftermedeltidaRegest=true&AvanceradSok=False&page=1&postid=sdhk_"
        url3 = "&tab=post#tab"
        url = url1 + str(number) + url2 + str(number) + url3
        response = requests.get(url)
        response_content = response.content.decode('utf-8')
        # Find place, language and content
        def getMetaDataFromTemplate(response_content, markerTemplate):
            # Find the first template occurence
            idx = response_content.find(markerTemplate)
            if idx >= 0:
                # Cut after the first occurence and before the second
                text = response_content[idx+len(markerTemplate):]
                text = text[:text.find(markerTemplate)]
                # Remove unwanted characters and tags
                removedSomething = True
                while removedSomething:
                    if text[0] in ['\r', '\n', ' ']:
                        text = text[1:]
                        removedSomething = True
                    else:
                        removedSomething = False
                    # Remove tags (not nested tags)
                    if text[0] == '<':
                        # Loop until the end ">"
                        while text[0] != '>':
                            text = text[1:]
                        text = text[1:]
                        removedSomething = True
                # Return data before next tag
                return text[:text.find('<')]
            else:
                return None
        # Find the date
        date_as_text = getMetaDataFromTemplate(response_content, """<h5>Datering</h5><span class="sdhk-brevhuvud">""")
        date_as_text = [s for s in date_as_text if s.isalnum() or s==' ']
        n = 1
        while n < len(date_as_text):
            if date_as_text[n-1].isspace() and date_as_text[n].isspace():
                date_as_text.pop(n)
            else:
                n += 1
        date_as_text = ''.join(date_as_text)
        if date_as_text is not None:
            year = int(re.findall(r'\d{4}', date_as_text)[0])
        else:
            year = None
        lang = getMetaDataFromTemplate(response_content, "<h5>Språk</h5><p>")
        place = getMetaDataFromTemplate(response_content, """<h5>Utfärdandeort</h5><span class="sdhk-brevhuvud">""")
        textcontent = getMetaDataFromTemplate(response_content, """<h5>Brevtext</h5><div class="sdhk-brevtext"><p>""")
    #            self._set(number, 'metadata_status_code', response.status_code)
    #            self._set(number, 'date_as_text', date_as_text)
    #            self._set(number, 'year', year)
    #            self._set(number, 'language', lang)
    #            self._set(number, 'origin', place)
    #            self._set(number, 'textcontent', textcontent)
        return_dict['metadata_status_code'] = response.status_code
        return_dict['date_as_text'] = date_as_text
        return_dict['year'] = year
        return_dict['language'] = lang
        return_dict['origin'] = place
        return_dict['textcontent'] = textcontent
        # Find url to printed text            
    #            markerTemplate = "<b>Tryckt</b>"
    #            idx = response.content.find(markerTemplate)
    #            printedurl = None
    #            if idx >= 0:
    #                text = response.content[idx+len(markerTemplate):]
    #                text = text[:text.find(markerTemplate)]
    #                match = re.search(r'href=[\'"]?([^\'" >]+)', text)
    #                if match:
    #                    printedurl = match.group(1)
    #            self._set(number, 'printedurl', printedurl)
    #            self._set(number, 'metadata_parsed', True)
        return_dict['exception'] = False
    except:
        return_dict['exception'] = True
    return return_dict

class SDHKHarvester():
    def __init__(self, savefile):
        self.reprname = "'Svenskt Diplomatariums Huvudkartotek'"
        # Load saved data or initialize
        self.savefile = savefile
        if os.path.exists(savefile):
            import json
            import gzip
            with gzip.open(savefile, 'r') as f:
                self.data = json.loads(f.read().decode('utf-8'))
        else:
            self.data = dict()
        self._good_ids = None

    def save(self):
        import gzip
        import json
        with gzip.open(self.savefile, 'w') as f:
            f.write(json.dumps(self.data, sort_keys=True, indent=2, 
                               separators=(',', ': ')).encode('utf-8'))
            
#    def _set(self, number, key, value):
#        if number in self.data.keys():
#            self.data[number][key] = value
#        else:
#            self.data[number] = {key : value}

#    def _dlLowDef(self, number):
#        url = "http://www3.ra.se/sdhk/bild/" + str(number) + ".JPG"
#        response = requests.get(url)
#        self._set(number, 'lowdef_status_code', response.status_code)
#        if response.status_code == 200:
#            f = open(os.path.join(self._lowDefPath, str(number) + ".jpg"), 'w')
#            f.write(response.content)
#            f.close()
#        return response.status_code

    def download(self, number):
        assert type(number) == type(list()) or type(number) == int
        if type(number) == type(list()):
            from multiprocessing import Pool
            with Pool(processes=os.cpu_count()*2) as pool:
                for d in pool.imap_unordered(download_metadata, number):
                    print("Downloading (using Pool) meta data for %i" % d['sdhk'])
                    if str(d['sdhk']) not in self.data.keys():
                        self.data[str(d['sdhk'])] = dict()
                    for k in d.keys():
                        self.data[str(d['sdhk'])][k] = d[k]
        else:
            print("Downloading meta data for %i" % d['sdhk'])
            d = download_metadata(number)
            if str(d['sdhk']) not in self.data.keys():
                self.data[str(d['sdhk'])] = dict()
            for k in d.keys():
                self.data[str(d['sdhk'])][k] = d[k]
        if self._good_ids is not None:
            self._update_good_ids()

    def get_good_ids(self):
        if self._good_ids is None:
            self._update_good_ids()
        import copy
        return copy.copy(self._good_ids)

    def _update_good_ids(self):
        self._good_ids = [int(k) for k in self.data.keys() 
            if not self.data[k]['exception'] and self.data[k]['metadata_status_code']==200]

    def keys(self):
        return [int(n) for n in self.data.keys()]
    
    def clear(self):
        self.data = dict()

    def __getitem__(self, number):
        import copy
        return copy.deepcopy(self.data[str(number)])

    def populate(self):
        self.download(list(range(1, 45000)))

    def _scan_path(self, basepath):
        assert os.path.exists(basepath)
        sdhk_ids = self.get_good_ids()
        found = dict()
        for (dirpath, dirnames, filenames) in os.walk(basepath):
            for filename in filenames:
                try:
                    fn = filename[:filename.find('.')]
                    sdhk_id = int(fn)
                    if str(sdhk_id) == fn and sdhk_id in sdhk_ids:
                        found[sdhk_id] = os.path.join(dirpath, filename)
                except:
                    pass
        return found
        
    def scan_highdef_path(self, basepath):
        d = self._scan_path(basepath)
        for k in d.keys():
            self.data[str(k)]['highdefpath'] = d[k]

    def scan_lowdef_path(self, basepath):
        d = self._scan_path(basepath)
        for k in d.keys():
            self.data[str(k)]['lowdefpath'] = d[k]

    def remove_bad_paths(self):
        for n in self.data.keys():
            for e in ['lowdefpath', 'highdefpath']:
                if e in self.data[n].keys() and not os.path.exists(self.data[n][e]):
                    self.data[n].pop(e)


if __name__ == '__main__':
    tmp_path = "~/tmp"
    lowdef_path = """/media/fredrik/UB Storage/Images/SDHK/LowDef"""
    highdef_path = """/media/fredrik/UB Storage/Images/SDHK/HighDef"""
    tmp_path  = os.path.expanduser(tmp_path)
    assert os.path.exists(tmp_path)
    savefile = os.path.join(tmp_path, "sdhk_metadata.json.gz")
    harvester = SDHKHarvester(savefile)
    # Popluate
    dl_keys = list(set(range(1, 50000)).difference(set(harvester.keys())))
    harvester.download(dl_keys)
    print("%i good ids in database" % (len(harvester.get_good_ids())))
    print("%i text entries in database" % 
          (np.sum([harvester[n]['textcontent'] is not None and len(harvester[n]['textcontent'])>50 for n in harvester.get_good_ids()])))
    if os.path.exists(lowdef_path):
        harvester.scan_lowdef_path(lowdef_path)
    print("%i low def images in database" % 
          (np.sum(['lowdefpath' in harvester[n] for n in harvester.get_good_ids()])))
    if os.path.exists(highdef_path):
        harvester.scan_highdef_path(highdef_path)
    print("%i high def images in database" % 
          (np.sum(['highdefpath' in harvester[n] for n in harvester.get_good_ids()])))
    harvester.save()
    
    #%% Plot histogram over text lengths
    text_ids = [n for n in harvester.get_good_ids() if harvester[n]['textcontent'] is not None]
    text_lengths = [len(harvester[n]['textcontent']) for n in text_ids]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title("Length of transcribed texts")
    plt.hist(text_lengths, 100)
    plt.xlabel('Lenght in characters')
    plt.ylabel('Number of documents')
    plt.xlim(np.min(text_lengths), np.max(text_lengths))
    plt.show()

    #%% Plot histogram of years for all entries in SDHK
    dated_ids = [n for n in harvester.get_good_ids() if 0 < harvester[n]['year'] and harvester[n]['year'] <= 1661]
    years = [harvester[n]['year']  for n in dated_ids]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title("SHDK documents per years")
    plt.hist(years, 200)
    plt.xlabel('Year')
    plt.ylabel('Number of documents')
#    plt.xlim(1135, 1546)
    plt.show()
#    plt.savefig("sdhk.pdf")
