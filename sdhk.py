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

# TODO Make handler class for json file
# TODO Make python3 friendly

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

    def keys(self):
        return [int(n) for n in self.data.keys()]
    
    def get_good_ids(self):
        return [int(k) for k in self.data.keys() 
                if not self.data[k]['exception'] and self.data[k]['metadata_status_code']==200]

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


#    def getGoodNumbers(self):
#        ret = []
#        for n in self._metadata.keys():
#            d = self._get(n)
#            keys = ['highdefpath', 'lowdefpath', 'year']
#            if np.all([k in d for k in keys]) and np.all([k is not None for k in keys]) and d['year'] > 1000:
#                ret.append(n)
#        return ret

#    def scanAndMarkBadImages(self):
#        # Find bad files
#        basepath = os.path.join(self._processedImagesPath, 'QualityClassification')
#        from os import listdir
#        badfiles = [f for f in listdir(os.path.join(basepath, 'Bad')) if os.path.isfile(os.path.join(basepath, 'Bad', f)) ]
#        # Find codes for files
#        badcodes = []
#        for filename in badfiles:
#            code = int(filename[:filename.find('.')])
#            badcodes.append(code)
#        # Set quality flag
#        for n in self.getNumbers():
#            if n in badcodes:
#                self._set(n, 'good_quality', False)
#            else:
#                self._set(n, 'good_quality', True)
#
#    def loadHighDefImage(self, number):
#        """Returns the high def image with the correct orientation. Some 
#        images are loaded strangly flipped."""
#        # Load image data
#        I = self._loadHighDefImageUnrotated(number)
#        J = self.loadLowDefImage(number)
#        Is = I.shape[0] < I.shape[1]
#        Js = J.shape[0] < J.shape[1]
#        if (Is and Js) or (not Is and not Js):
#            return I
#        else:
#            # Flip dims
#            return np.flipud(np.fliplr((I.transpose())))
#
#    def _loadHighDefImageUnrotated(self, number):
#        md = self._get(number)
#        assert md is not None, "Missing record " + str(number)
#        assert 'highdefpath' in md.keys(), "High def image path missing " + str(number)
#        name = md['highdefpath']
#        assert os.path.isfile(name), "Error high def in file name"
##        print("_loadHighDefImageUnrotated.name = %s" % name)
#        return self._loadCR2(name)
#
#    def _loadCR2(self, filename, convert2grayscale=True):
##        print("_loadCR2.filename = %s" % filename)
#        assert os.path.isfile(filename), "Error in CR2 file name (%s)" % filename
#        I = imageio.imread(filename)
#        I = np.asarray(I, dtype=np.float)
#        I *= 255/np.max(I.ravel())
#        I = np.asarray(I, dtype=np.uint8)
#        if convert2grayscale and I.ndim==3:
#            from skimage.color import rgb2gray
#            I = rgb2gray(I)
#        return I
#
#    def loadLowDefImage(self, number, colour=False):
#        md = self._get(number)
#        assert md is not None, "Missing record " + str(number)
#        assert 'lowdefpath'in md.keys(), "Low def image path missing " + str(number)
#        name = md['lowdefpath']
#        assert os.path.isfile(name), "Error in lowdef file name, not a file"
#        if colour:
#            return cv2.imread(name)
#        else:
#            return cv2.imread(name, cv2.IMREAD_GRAYSCALE)
#        
#    def _getSegmentedLetter(self, number):
#        M = self.loadHighDefImage(number)
#        # Do Otsu binarization of blurred image {0, 1}
#        blur = cv2.GaussianBlur(M, (5,5), 0)
#        thresh, bw = cv2.threshold(np.asarray(blur, dtype=np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        bw[bw>0]=1
#        # Find connected components and their areas
#        contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#        ccarea = [cv2.contourArea(c) for c in contours]
#        # Make mask image with only the lagest CC filled
#        N = np.zeros(M.shape, dtype=np.uint8)
#        N[:] = 255
#        cv2.fillPoly(N, contours[np.argmax(ccarea)], 0)
#        retval, rect = cv2.floodFill(image=N, mask=np.zeros((N.shape[0]+2, N.shape[1]+2), dtype=np.uint8), seedPoint=(0, 0), newVal=0, flags=cv2.FLOODFILL_FIXED_RANGE)
#        # Errode 
#        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
#        mask =  cv2.erode(N, kernel, iterations = 1)
#        # After erosion there might be disconnected components
#        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#        ccarea2 = [cv2.contourArea(c) for c in contours]
#        # Returns the bounding rectangle of the largest CC and mask
#        return cv2.boundingRect(contours[np.argmax(ccarea2)]), mask
#
#    def _getLessSegmentedLetter(self, number):
#        M = self.loadHighDefImage(number)
#        # Do Otsu binarization of blurred image {0, 1}
#        blur = cv2.GaussianBlur(M, (5,5), 0)
#        thresh, bw = cv2.threshold(np.asarray(blur, dtype=np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        bw[bw>0]=1
#        # Find connected components and their areas
#        _, contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#        ccarea = [cv2.contourArea(c) for c in contours]
#        # Select the largest CC as ROI
#        roi = np.asarray(cv2.boundingRect(contours[np.argmax(ccarea)]), dtype=np.int)
#        # Make mask
#        N = np.zeros(M.shape, dtype=np.uint8)
#        N[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = 1
#        return roi, N
#
#    def dataIterator(self, numbers, usecache=True, moreSegmentation=False):
#        assert not moreSegmentation, "Not implemented"
#        for number in numbers:
#            # Check for cached data, else generate data
#            cachedDataFileName = os.path.join(self._cachedDataPath, "cachedData_" + str(number) + ".npz")
#            if os.path.exists(cachedDataFileName) and usecache:# or (time.time() - os.path.getmtime(cachedDataFileName)) < 24*60*60:
#                data = np.load(cachedDataFileName)['data'].tolist()
#            else:
##                from ocropus_nlbin import binarize_image
#                GRAY = np.asarray(self.loadHighDefImage(number), dtype=np.uint8)
#                highdefshape = np.asarray([GRAY.shape[0], GRAY.shape[1]], dtype=np.int)
#                # Use only graylevels from segmented part
##                r, MASK = self._getSegmentedLetter(number)
#                r, MASK = self._getLessSegmentedLetter(number)
#                r = np.asarray(r, dtype=np.int)
#                # Initial Otsu
##                thresh, bw = cv2.threshold(GRAY[MASK>0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##                BW = GRAY.copy()
##                BW[GRAY<=thresh] = 255
##                BW[GRAY>thresh] = 0
#                # Only keep masked area
##                BW[MASK==0] = 0
#                MASK[MASK>0] = 1
#                # Crop  mask
##                tmp = MASK[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
##                for i in range(2):
##                    s = np.sum(tmp, axis=(i+1)%2) # Sum over "wrong" axis
##                    while s[-1] < (tmp.shape[(i+1)%2]*.3) and r[2] >= 500 and r[3] >= 500:
##                        s = s[:-1]
##                        if i == 0:
##                            r[3] -= 1
##                        else:
##                            r[2] -= 1
#                MASK = MASK[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
#                GRAY = GRAY[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
#                # Re-binarize using the new mask
#                # Otsu
##                thresh, bw = cv2.threshold(GRAY[MASK>0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##                BW = GRAY.copy()
##                BW[GRAY<=thresh] = 255
##                BW[GRAY>thresh] = 0
#                # Binarization from ocropus by Tomas Breul et al.
##                BW = np.asarray(1-binarize_image(np.asarray(GRAY.copy(), dtype=np.float64)), dtype=np.uint8)
#                # Mask BW
##                BW *= MASK
##                BW = self._clearMaskBorder(MASK, BW)
#                # Crop more at bottom
##                vertsum = np.sum(BW, axis=1)
##                cutoff = len(vertsum)
###                while cutoff >= 500 and np.max(vertsum[cutoff-30:]) < np.median(vertsum):
##                while cutoff >= 500 and np.max(vertsum[cutoff-80:]) < np.mean(vertsum):
##                    cutoff -= 20
##                GRAY = GRAY[:cutoff, :]
##                MASK = MASK[:cutoff, :]
##                BW = BW[:cutoff, :]
##                BW[BW>0] = 255
##                r[3] = cutoff
#                roi = np.asarray(r, dtype=np.int)
#                # Get low def data
#                GRAY_LOWDEF = self.loadLowDefImage(number, colour=False)
#                scaleFactor = np.asarray(highdefshape, dtype=np.float64)/np.asarray(GRAY_LOWDEF.shape, dtype=np.float64)
#                roi_lowdef = np.round(roi.copy() / scaleFactor[[0, 1, 0, 1]])
#                roi_lowdef = np.asarray(roi_lowdef, dtype=np.int)
#                GRAY_LOWDEF = GRAY_LOWDEF[roi_lowdef[1]:roi_lowdef[1]+roi_lowdef[3], roi_lowdef[0]:roi_lowdef[0]+roi_lowdef[2]]
#                MASK_LOWDEF = cv2.resize(MASK.copy(), (int(GRAY_LOWDEF.shape[1]), int(GRAY_LOWDEF.shape[0])), interpolation=cv2.INTER_NEAREST)
##                BW_LOWDEF = np.asarray(1-binarize_image(np.asarray(GRAY_LOWDEF.copy(), dtype=np.float64)), dtype=np.uint8)
##                BW_LOWDEF *= MASK_LOWDEF
##                BW_LOWDEF = self._clearMaskBorder(MASK_LOWDEF, BW_LOWDEF)
##                BW_LOWDEF[BW_LOWDEF>0] = 255
#                # Create return data dict
#                data = {'id': number,
##                        'bw': np.asarray(BW, dtype=np.uint8),
#                        'mask': np.asarray(MASK, dtype=np.uint8),
#                        'roi_lowdef': np.asarray(roi_lowdef, dtype=np.int),
##                        'bw_lowdef': np.asarray(BW_LOWDEF, dtype=np.uint8),
#                        'mask_lowdef': np.asarray(MASK_LOWDEF, dtype=np.uint8),
#                        'highdefshape': highdefshape,
#                        'roi': roi, 
#                        'date': self._get(number)['date_as_text'],
#                        'year': self._get(number)['year'],
#                        'origin': self._get(number)['origin']}
#                # Save to cache
#                np.savez_compressed(cachedDataFileName, data=data)
#            # Load gray scale data
#            roi = data['roi']
#            GRAY = np.asarray(self.loadHighDefImage(number), dtype=np.uint8)
#            data['gray'] = GRAY[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
#            # Get low def data
#            GRAY_LOWDEF = self.loadLowDefImage(number, colour=False)
#            roi_lowdef = np.asarray(data['roi_lowdef'], dtype=np.int)
#            GRAY_LOWDEF = GRAY_LOWDEF[roi_lowdef[1]:roi_lowdef[1]+roi_lowdef[3], roi_lowdef[0]:roi_lowdef[0]+roi_lowdef[2]]
#            data['gray_lowdef'] = np.asarray(GRAY_LOWDEF, dtype=np.uint8)
#            yield data

#    def _clearMaskBorder(self, MASK, BW):
#        """Clear border of MASK in BW"""
#        BW = BW.copy()
#        contours, hierarchy = cv2.findContours(MASK.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#        assert len(contours) != 0, "No mask"
#        if len(contours) > 1:
#            print("Multiple foreground areas in mask")
#        floodfillMask = np.zeros((BW.shape[0]+2, BW.shape[1]+2), dtype=np.uint8)
#        for p in contours[0]:
#            p1 = p[0][0]
#            p0 = p[0][1]
#            if BW[p0, p1] > 0:                        
#                retval, rect = cv2.floodFill(image=BW, mask=floodfillMask, seedPoint=(p1, p0), newVal=0, flags=cv2.FLOODFILL_FIXED_RANGE)
#        return BW


#    def _reinitEvenSample(self, N):
#        indices = self.getGoodNumbers()
#        years = [self._get(n)['year'] for n in indices]
#        binidx = np.asarray(np.floor(np.asarray(years)/10), dtype=np.int)
#        bins = {}
#        for b, i in zip(binidx, indices):
#            if b in bins:
#                bins[b].append(i)
#            else:
#                bins[b] = [i]
#        from random import shufflej
#        for k in bins.keys():
#            shuffle(bins[k])
#        self._evensample = []
#        while len(self._evensample) < N:
#            for k in bins.keys():
#                if len(self._evensample) < N and len(bins[k]) > 0:
#                    self._evensample.append(bins[k].pop())
#    
#    def getEvenSampleNumbers(self, N = 1000):
#        """Returns the id numbers of a sample of some size 
#        that is a evenly sampled in years as possible"""
#        if (self._evensample is not None and len(self._evensample) != N) or \
#                self._evensample is None:
#            self._reinitEvenSample(N)
#        # Return a copy
#        return list(self._evensample)

if __name__ == '__main__':
    tmp_path = "~/tmp"
    lowdef_path = """/media/fredrik/UB Storage/Images/SDHK/LowDef"""
    highdef_path = """/media/fredrik/UB Storage/Images/SDHK/HighDef"""
    tmp_path  = os.path.expanduser(tmp_path)
    assert os.path.exists(tmp_path)
    savefile = os.path.join(tmp_path, "sdhk_metadata.json")
    harvester = SDHKHarvester(savefile)
    # Popluate
    dl_keys = [n for n in range(1, 50000) if n not in harvester.keys()]
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
    plt.hist(text_lengths, 100)
    plt.xlim(np.min(text_lengths), np.max(text_lengths))
    plt.show()

    #%% Plot histogram of years for all entries in SDHK
    dated_ids = [n for n in harvester.get_good_ids() if 0 < harvester[n]['year'] and harvester[n]['year'] <= 1661]
    years = [harvester[n]['year']  for n in dated_ids]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.hist(years, 200)

    plt.xlabel('Year')
    plt.ylabel('Number of documents')
#    plt.xlim(1135, 1546)
    plt.show()
#    plt.savefig("sdhk.pdf")
