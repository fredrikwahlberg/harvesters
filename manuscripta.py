# -*- coding: utf-8 -*-
import requests
import re
import os.path
import json
import gzip
import copy
from multiprocessing import Pool
import os

# TODO Check for multiple language tags
# TODO Fix decoding bug

#MANIFEST_URLS = ["https://www.manuscripta.se/iiif/collection.json",
#"https://www.manuscripta.se/iiif/collection-ttt.json",
#"https://www.manuscripta.se/iiif/collection-greek.json"]

MANIFEST_URLS = ["https://www.manuscripta.se/iiif/collection-ttt.json"]

def download_image(data):
    url, filename = data
    import urllib.request
    try:
        urllib.request.urlretrieve(url, filename)
    except:
        pass
    if os.path.exists(filename):
        return os.path.getsize(filename)
    else:
        return 0

class Manuscripta:
    def __init__(self, basepath):
        self._basepath = basepath
        self._savefile = os.path.join(basepath, "metadata.json.gz")
        if os.path.exists(self._savefile):
            with gzip.open(self._savefile, 'r') as f:
                self._data = json.loads(f.read().decode('utf-8'))
        else:
            self._data = dict()
        self._manifests = None
        self._n_images = None

    def _webget(self, url):
        response = requests.get(url)
        assert response.ok, "Error reading url %s" % url
        try:
            return response.content.decode('utf-8', errors='replace')
        except:
            print("Error while decoding url %s" % url) 
            return None

    def _webget_json(self, url):
        try:
            return json.loads(self._webget(url))
        except:
            print("Error while JSON decoding url %s" % url) 
            return None

    def save(self):
        import gzip
        import json
        with gzip.open(self._savefile, 'w') as f:
            f.write(json.dumps(self._data, sort_keys=True, indent=2, 
                               separators=(',', ': ')).encode('utf-8'))

    def keys(self):
        return [int(n) for n in self._data.keys()]
    
    def clear(self):
        self._data = dict()

    def __repr__(self):
        return 'Manuscripta.se harvester with %i records' % len(self.keys())

    def __getitem__(self, number):
        import copy
        return copy.deepcopy(self._data[str(number)])

    def populate(self):
        self.load_manifests()
        for man in self.load_manifests():
            manifest_data = self._webget_json(man['@id'])
            if manifest_data is not None:
                collection_id = int(manifest_data['@id'].split('/')[-2])
                label = manifest_data['label']
                tei_xml = manifest_data['seeAlso']['@id']
                language = None
                shelfmark = None
                language = None
                title = None
                date = None
                support = None
                extent = None
                for d in manifest_data['metadata']:
                    if d['label'] == 'Shelfmark':
                        shelfmark = d['value']
                    if d['label'] == 'Language':
                        language = d['value']
                    if d['label'] == 'Title':
                        title = d['value']
                    if d['label'] == 'Date':
                        date = d['value']
                    if d['label'] == 'Support':
                        support = d['value']
                    if d['label'] == 'Extent':
                        extent = d['value']
                images = list()
                for canvas in manifest_data['sequences'][0]['canvases']:
                    images.append({'width':canvas['images'][0]['resource']['width'],
                                    'label':canvas['label'],
                                    'height':canvas['images'][0]['resource']['height'],
                                    'url':canvas['images'][0]['resource']['@id'],
                                    'filename':None})
                self._data[str(collection_id)] = {'label':label, 
                                                    'tei_xml':tei_xml,
                                                    'language':language,
                                                    'shelfmark':shelfmark,
                                                    'language':language,
                                                    'title':title,
                                                    'date':date,
                                                    'support':support,
                                                    'extent':extent,
                                                    'images':images}
        self.check_filenames()

    def load_manifests(self):
        self._manifests = list()
        for man_url in MANIFEST_URLS:
            self._manifests.extend(self._webget_json(man_url)['manifests'])
        return self._manifests

    @property
    def manifest_labels_(self):
        if self._manifests is None:
            self.load_manifests()
        ret = list()
        for m in self._manifests:
            ret.append(copy.copy(m['label']))
        return ret

    @property
    def n_images_(self):
        if self._n_images is None:
            self.check_filenames()
        return self._n_images

    @property
    def n_bytes_(self):
        ret = os.path.getsize(self._savefile)
        for k in self._data.keys():
            for i in range(len(self._data[k]['images'])):
                filename = self._data[k]['images'][i]['filename']
                if filename is not None and os.path.exists(filename):
                    ret += os.path.getsize(filename)
        return ret

    def check_filenames(self):
        self._n_errors = 0
        self._n_images = 0
        download_list = list()
        for k in self._data.keys():
            for i in range(len(self._data[k]['images'])):
                url = self._data[k]['images'][i]['url']
                filename = self._data[k]['images'][i]['filename']
                if filename is not None and os.path.exists(filename):
                    self._n_images += 1
                else:
                    self._data[k]['images'][i]['filename'] = None
                    fn = next(x for x in url.split('/') if len(x)>3 and x[-3:]=='tif')
                    fn = fn[:-3] + "jpg"
                    filepath = os.path.join(self._basepath, str(k), fn)
                    if os.path.exists(filepath) and os.path.isfile(filepath):
                        self._data[k]['images'][i]['filename'] = filepath
                        self._n_images += 1
                    else:
                        download_list.append((url, filepath))
                    self._n_errors += 1
        return download_list

    def download_images(self, download_list=None):
        if download_list is None:
            download_list= self.check_filenames()
        for e in download_list:
            dirname = os.path.dirname(e[1])
            if not (os.path.exists(dirname) and os.path.isdir(dirname)):
                os.makedirs(dirname)
        with Pool(processes=os.cpu_count()-1) as pool:
            dl_bytes = 0
            for i, nbytes in enumerate(pool.imap_unordered(download_image, download_list)):
                dl_bytes += nbytes
                if i%10==0:
                    print("Downloaded %i files of %i, %.1f Mb" % (i, len(download_list), dl_bytes/1000000))
        self.check_filenames()

if __name__=='__main__':
    BASEPATH = os.path.expanduser("~/Data/Manuscripta")
    assert os.path.exists(BASEPATH) and os.path.isdir(BASEPATH)
    manuscripta = Manuscripta(BASEPATH)
#    manuscripta.populate()

    manuscripta.download_images()
#    download_list = manuscripta.check_filenames()
#    import random
#    random.shuffle(download_list)
#    dl_bytes = 0
#    for i, d in enumerate(download_list):
#        download_image(d)
#        dl_bytes += os.path.getsize(d[1])
#        if i%10==0:
#            print("Downloaded %i files of %i, %.1f Mb" % (i, len(download_list), dl_bytes/1000000))
    manuscripta.save()
    
    print(manuscripta)
    print("%i image, %.1f Mb" % (manuscripta.n_images_, manuscripta.n_bytes_/1e6))

    dates = [manuscripta[k]['date'] for k in manuscripta.keys()]
    languages = [manuscripta[k]['language'] for k in manuscripta.keys()]

#    url = "https://www.manuscripta.se/iiif/100120/manifest.json"
#    response = requests.get(url)
#    text = response.content.decode('utf-8')
#    json_dict = json.loads(text)


#ms_ids = [int(ms) for ms in re.findall(r'(?i)<a href="\/ms\/([^>]+)">', 
#                                          webget(browse_url))]
#
#tei_urls = ["https://www.manuscripta.se/xml/" + str(n) for n in ms_ids]
#
#t = tei_xml[160]
#
##"""<primaryAddress>[\s\S]*?<\/primaryAddress>"""
#
##re.findall(r'(?i)<surface ([^>]+)>(.+?)</surface>', t)
#surfaces = [t[1] for t in re.findall(r'(?i)<surface ([^>]+?)>([\s\S]+?)</surface>', tei_xml[160])]
#desc = [re.findall(r'(?i)<desc>(.+?)</desc>', s)[0] for s in surfaces]
#g = [re.findall(r'(?i)<graphic ([^>]+)/>', s)[0] for s in surfaces]
#o = g[0].split(' ')
##p = dict()
#for p in o:
#    k, v = p.split('=')
#    v = v.replace('"', '')
#    
#
#def image_url(number, filename, width):
#    return """https://www.manuscripta.se/iipsrv/iipsrv.fcgi?IIIF=""" + str(number) + """/""" + filename + """/full/""" + str(width) + """,/0/default.jpg"""
#
#image_url(100379, "uub-b-023_0003_001r.tif", 3530)
#
##%% lang
#surfaces = [t[1] for t in re.findall(r'(?i)<surface ([^>]+?)>([\s\S]+?)</surface>', tei_xml[160])]
#languages = [re.findall(r'(?i)<textLang mainLang="([^>]+)"/>', text) for text in tei_xml]

#textLang mainLang=""
