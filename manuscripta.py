# -*- coding: utf-8 -*-
import urllib.request
import re
import os.path
import json
import gzip
import copy
from multiprocessing import Pool

# TODO Fix decoding bug
# TODO Very little debugging/testing is done

BASEURL = """https://www.manuscripta.se/iiif/"""
manifest_url = lambda n: BASEURL + str(n) + """/manifest.json"""

def _download_image(data):
    """Downloads an image to the database"""
    url, filename = data
    try:
        urllib.request.urlretrieve(url, filename)
    except:
        pass
    if os.path.exists(filename):
        return os.path.getsize(filename)
    else:
        return 0

def _webget(url):
    """Fetches and decodes text files from the internet"""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read().decode('utf-8', errors='replace')
        return data
    except:
        print("Error while getting or decoding data from url %s" % url)
        return None

def _webget_json(url):
    """Fetches and decodes JSON files from the internet"""
    try:
        return json.loads(_webget(url))
    except:
        print("Error while decoding JSON from url %s" % url) 
        return None

class Manuscripta:
    def __init__(self, basepath, verbose=True):
        self._basepath = basepath
        self.verbose = verbose
        self._savefile = os.path.join(basepath, "metadata.json.gz")
        if os.path.exists(self._savefile):
            with gzip.open(self._savefile, 'r') as f:
                self._data = json.loads(f.read().decode('utf-8'))
        else:
            self._data = dict()
        # self._manifests = None
        self._n_images = None

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
        numbers = self._download_manifest_numbers()
        for manifest_data in self._download_manifests(numbers):
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
                for metadata_item in manifest_data['metadata']:
                    if metadata_item['label'].lower().find("shelfmark") >= 0:
                        shelfmark = metadata_item['value']
                    if metadata_item['label'].lower().find("language") >= 0:
                        language = metadata_item['value']
                    if metadata_item['label'].lower().find("title") >= 0:
                        title = metadata_item['value']
                    if metadata_item['label'].lower().find("date") >= 0:
                        date = metadata_item['value']
                    if metadata_item['label'].lower().find("support") >= 0:
                        support = metadata_item['value']
                    if metadata_item['label'].lower().find("extent") >= 0:
                        extent = metadata_item['value']
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

    def _download_manifest_numbers(self):
        """Downloads the numbers of all manifests"""
        # Loading manifest list
        with urllib.request.urlopen(BASEURL) as response:
            base_xml = response.read().decode()
        # Parse manufest numbers
        p = r'ion name=\"(\d+)\"'
        manifest_numbers = re.findall(p, base_xml)
        assert type(manifest_numbers) is list
        return manifest_numbers

    def _download_manifests(self, numbers):
        """Downloads manifests with numbers given as a list"""
        assert type(numbers) is list
        # Make urls to download
        manifest_urls = list(map(manifest_url, numbers))
        # Download in parallel
        parsed_manifests = list()
        with Pool(processes=max(os.cpu_count()*10, 20)) as pool:
        #with Pool(40) as pool:
            for data in pool.imap_unordered(_webget_json, manifest_urls):
                parsed_manifests.append(data)
                if data is not None and self.verbose:
                    print("Decoded", data['@id'])
        return parsed_manifests

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
        """Runs an inventory over the image files
        Returns a list over missing files and url and local file name"""
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
        with Pool(processes=min(os.cpu_count()*4, 10)) as pool:
            dl_bytes = 0
            for i, nbytes in enumerate(pool.imap_unordered(_download_image, download_list)):
                dl_bytes += nbytes
                if (i%10) == 9:
                    print("Downloaded %i files of %i, %.1f Mb" % (i+1, len(download_list), dl_bytes/1000000))
        self.check_filenames()

if __name__=='__main__':
    BASEPATH = os.path.expanduser("~/Data/Manuscripta")
    assert os.path.exists(BASEPATH) and os.path.isdir(BASEPATH)
    manuscripta = Manuscripta(BASEPATH)
    
    manuscripta.populate()
    manuscripta.save()

    dates = [manuscripta[k]['date'] for k in manuscripta.keys()]
    languages = [manuscripta[k]['language'] for k in manuscripta.keys()]
    isSwedish = lambda lang: lang.lower().find("swe")>=0 or lang.lower().find("sv")>=0
    in_swedish = [k for k in manuscripta.keys() if isSwedish(manuscripta[k]['language'])]
    print("%i manuscripts in swedish" % len(in_swedish))

    # Download images from swedish sources
    n_images = 50
    if manuscripta.n_images_ < n_images :
        download_list = manuscripta.check_filenames()
        download_list = [e for e in download_list if int(e[1].split("/")[-2]) in in_swedish]
        import random
        random.shuffle(download_list)
        download_list = download_list[:n_images-manuscripta.n_images_]
        manuscripta.download_images(download_list)
        manuscripta.save()
    
    print(manuscripta)
    print("DB contains %i image, %.1f Mb" % (manuscripta.n_images_, manuscripta.n_bytes_/1e6))

