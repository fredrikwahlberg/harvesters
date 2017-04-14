# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 16:30:33 2014

@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

import requests
import os.path
import re
import numpy as np
import cv2
import imageio

if __name__ == '__main__':
    from base import DataLoaderBase
else:
    from .base import DataLoaderBase

class SdhkLoader(DataLoaderBase):
    def __init__(self, saveFilePath, highDefPath=None, lowDefPath=None, cachedDataPath=None):
        super(SdhkLoader, self).__init__()
        self.reprname = "'Svenskt Diplomatariums Huvudkartotek'"
        # Load saved data or initialize
        self._metadata = {}
        self._evensample = None
        self._paths = None
        if os.path.isfile(saveFilePath):
            data = np.load(saveFilePath)
            self._metadata = data['metadata'].tolist()
            for key in data.iterkeys():
                if key == 'evensample': # Hack to replace has_key
                    self._evensample = data['evensample'].tolist()
                if key == 'paths': # Hack to replace has_key
                    self._paths = data['paths'].tolist()
        if self._paths is None:
            self._paths = {'highDefPath': highDefPath, 
                           'lowDefPath': lowDefPath,
                           'cachedDataPath': cachedDataPath}
        self._saveFilePath = saveFilePath
#        assert os.path.isfile(self._saveFilePath), "Can't find db file"
        if highDefPath is None:
            self._highDefPath = self._paths['highDefPath']
        else:
            self._highDefPath = highDefPath
            self._paths['highDefPath'] = self._highDefPath
        if lowDefPath is None:
            self._lowDefPath = self._paths['lowDefPath']
        else:
            self._lowDefPath = lowDefPath
            self._paths['lowDefPath'] = self._lowDefPath
        if cachedDataPath is None:
            self._cachedDataPath = self._paths['cachedDataPath']
        else:
            self._cachedDataPath = cachedDataPath
            self._paths['cachedDataPath'] = self._cachedDataPath
        assert os.path.isdir(self._highDefPath)
        assert os.path.isdir(self._lowDefPath)
        assert os.path.isdir(self._cachedDataPath)

    def _dlLowDef(self, number):
        url = "http://www3.ra.se/sdhk/bild/" + str(number) + ".JPG"
        response = requests.get(url)
        self._set(number, 'lowdef_status_code', response.status_code)
        if response.status_code == 200:
            f = open(os.path.join(self._lowDefPath, str(number) + ".jpg"), 'w')
            f.write(response.content)
            f.close()
        return response.status_code
    
    def _dlMetaData(self, number):
        try:
            print("Downloading meta data for %i" % number)
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
            self._set(number, 'metadata_status_code', response.status_code)
            self._set(number, 'date_as_text', date_as_text)
            self._set(number, 'year', year)
            self._set(number, 'language', lang)
            self._set(number, 'origin', place)
            self._set(number, 'textcontent', textcontent)
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
            self._set(number, 'metadata_parsed', True)
        except:            
            self._set(number, 'metadata_parsed', False)
            print("Error while downloading or parsing metadata for SDHK nr. " + str(number))

    def save(self):
        np.savez_compressed(self._saveFilePath, metadata=self._metadata, 
                            evensample=self._evensample, paths=self._paths)

    def clear(self):
        self._metadata = {}
        self._evensample = None

    def _get(self, number):
        """Returns shallow copy of the data post"""
        if number in self._metadata.keys():
            # Ensure that all tags exist
            keys = ['date_as_text', 'year', 'origin', 'language', 'textcontent', 'metadata_status_code', 'metadata_parsed']
            getMetatdata = np.any([k not in self._metadata[number].keys() for k in keys])
            # Filter also in the value of some tags
            if not getMetatdata:
                getMetatdata = self._metadata[number]['metadata_status_code'] != 200 or \
                                    not self._metadata[number]['metadata_parsed']
            # Download meta data
            if getMetatdata:
                self._dlMetaData(number)
            # Return shallow copy
            return self._metadata[number].copy()
        else:
            return None

    def __getitem__(self, number):
        import copy
        return copy.deepcopy(self._get(number))

    def getNumbers(self):
        return self._metadata.keys()

    def getGoodNumbers(self):
        ret = []
        for n in self._metadata.keys():
            d = self._get(n)
            keys = ['highdefpath', 'lowdefpath', 'year']
            if np.all([k in d for k in keys]) and np.all([k is not None for k in keys]) and d['year'] > 1000:
                ret.append(n)
        return ret

    def _set(self, number, key, value):
        if number in self._metadata.keys():
            self._metadata[number][key] = value
        else:
            self._metadata[number] = {key : value}

    def _has_key(self, number, key):
        if number in self._metadata.keys() and key in self._metadata[number].keys():
            return True
        return False

    def populate(self):
        # TODO: Debug
        sdhk.scanHighDefPath()
        sdhk.scanLowDefPath()
#        sdhk.scanAndMarkBadImages()
        
    def scanHighDefPath(self):
        from os import walk
        found = []
        # Traverse path and add entries to db
        for (dirpath, dirnames, filenames) in walk(self._highDefPath):
            for filename in filenames:
                imageFileName = os.path.join(dirpath, filename)
#                print imageFileName
                code = filename[:filename.find('.')]
                try:
                    code = int(code)
                    self._set(code, 'highdefpath', imageFileName)
                    found.append(int(code))
                except:
                    pass
        # Remove keys if path already in db was not found
        for n in self._metadata.keys():
            if n not in found:
                if 'highdefpath' in self._metadata[n].keys():
                    self._metadata[n].pop('highdefpath')

    def scanLowDefPath(self):
        from os import walk
        found = []
        # Traverse path and add entries to db
        for (dirpath, dirnames, filenames) in walk(self._lowDefPath):
            for filename in filenames:
                imageFileName = os.path.join(dirpath, filename)
                code = filename[:filename.find('.')]
                self._set(int(code), 'lowdefpath', imageFileName)
                found.append(int(code))
        # Remove keys if path already in db was not found
        for n in self._metadata.keys():
            if n not in found:
                if 'lowdefpath' in self._metadata[n].keys():
                    self._metadata[n].pop('lowdefpath')
        
    def scanAndMarkBadImages(self):
        # Find bad files
        basepath = os.path.join(self._processedImagesPath, 'QualityClassification')
        from os import listdir
        badfiles = [f for f in listdir(os.path.join(basepath, 'Bad')) if os.path.isfile(os.path.join(basepath, 'Bad', f)) ]
        # Find codes for files
        badcodes = []
        for filename in badfiles:
            code = int(filename[:filename.find('.')])
            badcodes.append(code)
        # Set quality flag
        for n in self.getNumbers():
            if n in badcodes:
                self._set(n, 'good_quality', False)
            else:
                self._set(n, 'good_quality', True)

    def loadHighDefImage(self, number):
        """Returns the high def image with the correct orientation. Some 
        images are loaded strangly flipped."""
        # Load image data
        I = self._loadHighDefImageUnrotated(number)
        J = self.loadLowDefImage(number)
        Is = I.shape[0] < I.shape[1]
        Js = J.shape[0] < J.shape[1]
        if (Is and Js) or (not Is and not Js):
            return I
        else:
            # Flip dims
            return np.flipud(np.fliplr((I.transpose())))

    def _loadHighDefImageUnrotated(self, number):
        md = self._get(number)
        assert md is not None, "Missing record " + str(number)
        assert 'highdefpath' in md.keys(), "High def image path missing " + str(number)
        name = md['highdefpath']
        assert os.path.isfile(name), "Error high def in file name"
#        print("_loadHighDefImageUnrotated.name = %s" % name)
        return self._loadCR2(name)

    def _loadCR2(self, filename, convert2grayscale=True):
#        print("_loadCR2.filename = %s" % filename)
        assert os.path.isfile(filename), "Error in CR2 file name (%s)" % filename
        I = imageio.imread(filename)
        I = np.asarray(I, dtype=np.float)
        I *= 255/np.max(I.ravel())
        I = np.asarray(I, dtype=np.uint8)
        if convert2grayscale and I.ndim==3:
            from skimage.color import rgb2gray
            I = rgb2gray(I)
        return I

    def loadLowDefImage(self, number, colour=False):
        md = self._get(number)
        assert md is not None, "Missing record " + str(number)
        assert 'lowdefpath'in md.keys(), "Low def image path missing " + str(number)
        name = md['lowdefpath']
        assert os.path.isfile(name), "Error in lowdef file name, not a file"
        if colour:
            return cv2.imread(name)
        else:
            return cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        
    def _getSegmentedLetter(self, number):
        M = self.loadHighDefImage(number)
        # Do Otsu binarization of blurred image {0, 1}
        blur = cv2.GaussianBlur(M, (5,5), 0)
        thresh, bw = cv2.threshold(np.asarray(blur, dtype=np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bw[bw>0]=1
        # Find connected components and their areas
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ccarea = [cv2.contourArea(c) for c in contours]
        # Make mask image with only the lagest CC filled
        N = np.zeros(M.shape, dtype=np.uint8)
        N[:] = 255
        cv2.fillPoly(N, contours[np.argmax(ccarea)], 0)
        retval, rect = cv2.floodFill(image=N, mask=np.zeros((N.shape[0]+2, N.shape[1]+2), dtype=np.uint8), seedPoint=(0, 0), newVal=0, flags=cv2.FLOODFILL_FIXED_RANGE)
        # Errode 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        mask =  cv2.erode(N, kernel, iterations = 1)
        # After erosion there might be disconnected components
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ccarea2 = [cv2.contourArea(c) for c in contours]
        # Returns the bounding rectangle of the largest CC and mask
        return cv2.boundingRect(contours[np.argmax(ccarea2)]), mask

    def _getLessSegmentedLetter(self, number):
        M = self.loadHighDefImage(number)
        # Do Otsu binarization of blurred image {0, 1}
        blur = cv2.GaussianBlur(M, (5,5), 0)
        thresh, bw = cv2.threshold(np.asarray(blur, dtype=np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bw[bw>0]=1
        # Find connected components and their areas
        _, contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ccarea = [cv2.contourArea(c) for c in contours]
        # Select the largest CC as ROI
        roi = np.asarray(cv2.boundingRect(contours[np.argmax(ccarea)]), dtype=np.int)
        # Make mask
        N = np.zeros(M.shape, dtype=np.uint8)
        N[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = 1
        return roi, N

    def dataIterator(self, numbers, usecache=True, moreSegmentation=False):
        assert not moreSegmentation, "Not implemented"
        for number in numbers:
            # Check for cached data, else generate data
            cachedDataFileName = os.path.join(self._cachedDataPath, "cachedData_" + str(number) + ".npz")
            if os.path.exists(cachedDataFileName) and usecache:# or (time.time() - os.path.getmtime(cachedDataFileName)) < 24*60*60:
                data = np.load(cachedDataFileName)['data'].tolist()
            else:
#                from ocropus_nlbin import binarize_image
                GRAY = np.asarray(self.loadHighDefImage(number), dtype=np.uint8)
                highdefshape = np.asarray([GRAY.shape[0], GRAY.shape[1]], dtype=np.int)
                # Use only graylevels from segmented part
#                r, MASK = self._getSegmentedLetter(number)
                r, MASK = self._getLessSegmentedLetter(number)
                r = np.asarray(r, dtype=np.int)
                # Initial Otsu
#                thresh, bw = cv2.threshold(GRAY[MASK>0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#                BW = GRAY.copy()
#                BW[GRAY<=thresh] = 255
#                BW[GRAY>thresh] = 0
                # Only keep masked area
#                BW[MASK==0] = 0
                MASK[MASK>0] = 1
                # Crop  mask
#                tmp = MASK[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
#                for i in range(2):
#                    s = np.sum(tmp, axis=(i+1)%2) # Sum over "wrong" axis
#                    while s[-1] < (tmp.shape[(i+1)%2]*.3) and r[2] >= 500 and r[3] >= 500:
#                        s = s[:-1]
#                        if i == 0:
#                            r[3] -= 1
#                        else:
#                            r[2] -= 1
                MASK = MASK[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
                GRAY = GRAY[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
                # Re-binarize using the new mask
                # Otsu
#                thresh, bw = cv2.threshold(GRAY[MASK>0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#                BW = GRAY.copy()
#                BW[GRAY<=thresh] = 255
#                BW[GRAY>thresh] = 0
                # Binarization from ocropus by Tomas Breul et al.
#                BW = np.asarray(1-binarize_image(np.asarray(GRAY.copy(), dtype=np.float64)), dtype=np.uint8)
                # Mask BW
#                BW *= MASK
#                BW = self._clearMaskBorder(MASK, BW)
                # Crop more at bottom
#                vertsum = np.sum(BW, axis=1)
#                cutoff = len(vertsum)
##                while cutoff >= 500 and np.max(vertsum[cutoff-30:]) < np.median(vertsum):
#                while cutoff >= 500 and np.max(vertsum[cutoff-80:]) < np.mean(vertsum):
#                    cutoff -= 20
#                GRAY = GRAY[:cutoff, :]
#                MASK = MASK[:cutoff, :]
#                BW = BW[:cutoff, :]
#                BW[BW>0] = 255
#                r[3] = cutoff
                roi = np.asarray(r, dtype=np.int)
                # Get low def data
                GRAY_LOWDEF = self.loadLowDefImage(number, colour=False)
                scaleFactor = np.asarray(highdefshape, dtype=np.float64)/np.asarray(GRAY_LOWDEF.shape, dtype=np.float64)
                roi_lowdef = np.round(roi.copy() / scaleFactor[[0, 1, 0, 1]])
                roi_lowdef = np.asarray(roi_lowdef, dtype=np.int)
                GRAY_LOWDEF = GRAY_LOWDEF[roi_lowdef[1]:roi_lowdef[1]+roi_lowdef[3], roi_lowdef[0]:roi_lowdef[0]+roi_lowdef[2]]
                MASK_LOWDEF = cv2.resize(MASK.copy(), (int(GRAY_LOWDEF.shape[1]), int(GRAY_LOWDEF.shape[0])), interpolation=cv2.INTER_NEAREST)
#                BW_LOWDEF = np.asarray(1-binarize_image(np.asarray(GRAY_LOWDEF.copy(), dtype=np.float64)), dtype=np.uint8)
#                BW_LOWDEF *= MASK_LOWDEF
#                BW_LOWDEF = self._clearMaskBorder(MASK_LOWDEF, BW_LOWDEF)
#                BW_LOWDEF[BW_LOWDEF>0] = 255
                # Create return data dict
                data = {'id': number,
#                        'bw': np.asarray(BW, dtype=np.uint8),
                        'mask': np.asarray(MASK, dtype=np.uint8),
                        'roi_lowdef': np.asarray(roi_lowdef, dtype=np.int),
#                        'bw_lowdef': np.asarray(BW_LOWDEF, dtype=np.uint8),
                        'mask_lowdef': np.asarray(MASK_LOWDEF, dtype=np.uint8),
                        'highdefshape': highdefshape,
                        'roi': roi, 
                        'date': self._get(number)['date_as_text'],
                        'year': self._get(number)['year'],
                        'origin': self._get(number)['origin']}
                # Save to cache
                np.savez_compressed(cachedDataFileName, data=data)
            # Load gray scale data
            roi = data['roi']
            GRAY = np.asarray(self.loadHighDefImage(number), dtype=np.uint8)
            data['gray'] = GRAY[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            # Get low def data
            GRAY_LOWDEF = self.loadLowDefImage(number, colour=False)
            roi_lowdef = np.asarray(data['roi_lowdef'], dtype=np.int)
            GRAY_LOWDEF = GRAY_LOWDEF[roi_lowdef[1]:roi_lowdef[1]+roi_lowdef[3], roi_lowdef[0]:roi_lowdef[0]+roi_lowdef[2]]
            data['gray_lowdef'] = np.asarray(GRAY_LOWDEF, dtype=np.uint8)
            yield data

    def _clearMaskBorder(self, MASK, BW):
        """Clear border of MASK in BW"""
        BW = BW.copy()
        contours, hierarchy = cv2.findContours(MASK.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(contours) != 0, "No mask"
        if len(contours) > 1:
            print("Multiple foreground areas in mask")
        floodfillMask = np.zeros((BW.shape[0]+2, BW.shape[1]+2), dtype=np.uint8)
        for p in contours[0]:
            p1 = p[0][0]
            p0 = p[0][1]
            if BW[p0, p1] > 0:                        
                retval, rect = cv2.floodFill(image=BW, mask=floodfillMask, seedPoint=(p1, p0), newVal=0, flags=cv2.FLOODFILL_FIXED_RANGE)
        return BW

    def _reinitEvenSample(self, N):
        indices = self.getGoodNumbers()
        years = [self._get(n)['year'] for n in indices]
        binidx = np.asarray(np.floor(np.asarray(years)/10), dtype=np.int)
        bins = {}
        for b, i in zip(binidx, indices):
            if b in bins:
                bins[b].append(i)
            else:
                bins[b] = [i]
        from random import shuffle
        for k in bins.keys():
            shuffle(bins[k])
        self._evensample = []
        while len(self._evensample) < N:
            for k in bins.keys():
                if len(self._evensample) < N and len(bins[k]) > 0:
                    self._evensample.append(bins[k].pop())
    
    def getEvenSampleNumbers(self, N = 1000):
        """Returns the id numbers of a sample of some size 
        that is a evenly sampled in years as possible"""
        if (self._evensample is not None and len(self._evensample) != N) or \
                self._evensample is None:
            self._reinitEvenSample(N)
        # Return a copy
        return list(self._evensample)

    
if __name__ == '__main__':
    # Set up loader
    dbPath = "/home/fredrik/tmp/SDHK/"
    assert os.path.exists(dbPath), "Can't find path"
    saveFilePath = os.path.join(dbPath, "metadata.npz")
    highDefPath = "/media/fredrik/UB Storage/Images/SDHK/HighDef"
    lowDefPath = "/media/fredrik/UB Storage/Images/SDHK/LowDef"
    cachedDataPath = os.path.join(dbPath, "cache")
    sdhk = SdhkLoader(saveFilePath=saveFilePath, highDefPath=highDefPath, 
                      lowDefPath=lowDefPath, cachedDataPath=cachedDataPath)
    
    # Popluate
    sdhk.populate()
    sdhk.save()

    print("Checking (and downloading) meta data")
    for i, n in enumerate(list(sdhk._metadata.keys())):
        # Download meta data if needed
        d = sdhk[n]
        if i > 0 and i % 200 == 0:
            print("%i%% finished " % int(i*100/len(sdhk._metadata.keys())))
            sdhk.save()
    sdhk.save()


#    print("Loading images so thet segmentations are created")
#    numbers = sdhk.getGoodNumbers()
#    for d in sdhk.dataIterator(numbers):
#        pass

    # Plot histogram of years among the good documents
#    dates = []
#    index = []
#    haveTranscription = 0
#    for n in sdhk.getGoodNumbers():
#        d = sdhk._get(n)
#        dates.append(d['date']/10000)
#        index.append(n)
#        if d['textcontent'] is not None:
#            haveTranscription += 1
#
#    import matplotlib.pyplot as plt
#    plt.figure()
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='serif')
#
#    plt.hist(dates, 40)
#
#    dates2 = []
#    index2 = []
#    haveTranscription2 = 0
#    for n in sdhk.getEvenSampleNumbers()[:500]:
#        d = sdhk._get(n)
#        dates2.append(d['date']/10000)
#        index2.append(n)
#        if d['textcontent'] is not None:
#            haveTranscription2 += 1
#
#    plt.hist(dates2, 40)
#    plt.xlabel('Year')
#    plt.ylabel('Number of documents')
#    plt.xlim(1135, 1546)
#    plt.show()
##    plt.savefig("sdhk.pdf")
    
#    # Benchmark generator
#    N = 10
#    numbers = sdhk.getEvenSampleNumbers()[:N]
#    t = time.time()
#    for data in sdhk.imageGenerator(numbers):
#        pass
#    print N, "entries processed by the generator in", int(time.time() - t), "seconds ->", (time.time() - t)/float(N), "seconds/entry"
#    t = time.time()
#    for n in numbers:
#        GRAY = np.asarray(sdhk.loadHighDefImage(n), dtype=np.uint8)
#    print N, "images loaded in", int(time.time() - t), "seconds ->", (time.time() - t)/float(N), "seconds/image"

    # Load images through generator to ensure caching
#    for data in sdhk.imageGenerator(sdhk.getEvenSampleNumbers()):
#        print "Loaded id", data['id']

#    d = sdhk.imageGenerator(sdhk.getEvenSampleNumbers()[:1]).next()
#    from StringIO import StringIO
#    s = StringIO()
#    np.savez_compressed(s, data=d)
#    c = s.getvalue()
#    s.close()
#    from sys import getsizeof
#    print getsizeof(d), "->", getsizeof(c)
    
    # Close object and discard memory buffer --
    # .getvalue() will now raise an exception.
#    output.close()



#    nums = sdhk.getEvenSampleNumbers()
#    for i, data in enumerate(sdhk.imageGenerator(nums)):
#        print i, ":", data['id']

#    from random import sample
#    n = sample(sdhk.getGoodNumbers(), 6)
#
#    n = [32424, 24850, 10440, 20698, 5745, 28058]
##    n = [9115, 222, 33002, 615, 4867, 180]
#    plt.figure()
#    for i, d in enumerate(sdhk.imageGenerator(n)):
#        plt.subplot(3, 6, i*3+1)
#        plt.title("id: " + str(d['id']))
#        plt.imshow(d['gray'], cmap='gray')
#        plt.subplot(3, 6, i*3+2)
#        plt.imshow(d['bw'], cmap='gray')
#        plt.subplot(3, 6, i*3+3)
#        plt.imshow(d['mask'], cmap='gray')
#    plt.show()

#    origins = {}    
#    for n in sdhk.getGoodNumbers():
#        origin = sdhk._get(n)['origin']
#        if not origins.has_key(origin):
#            origins[origin] = 1
#        else:
#            origins[origin] += 1
#    origins.pop(None)
#    o = origins.keys()
#    l = []
#    n = origins.values()
#    for i in np.argsort(n):
#        l.insert(0, o[i])
#    n.sort(reverse=True)
#    plt.plot(np.cumsum(n)/float(np.sum(n)))
    
#    import time
#    n = [2105]
#    n = [32424, 24850, 10440, 20698, 5745, 28058, 9115, 222, 33002, 615, 4867, 180]
#    n = [24850]
#    n = [11190]
##    n = [1388, 1723, 1943, 2708, 3604, 4535, 5723]
#    import time
#    gene = sdhk.imageGenerator(n, usecache=False)
##    gene = sdhk.imageGenerator(n, usecache=True)
#    t = time.time()
#    data = gene.next()
#    print time.time()-t
#    import matplotlib.pyplot as plt
#    plt.figure()
#    plt.subplot(2, 3, 1)
#    plt.title('lowdef')
#    plt.imshow(data['gray_lowdef'], cmap='gray', interpolation='none')
#    plt.subplot(2, 3, 2)
#    plt.imshow(data['mask_lowdef'], cmap='gray', interpolation='none')
#    plt.subplot(2, 3, 3)
#    plt.imshow(data['bw_lowdef'], cmap='gray', interpolation='none')
#    plt.subplot(2, 3, 4)
#    plt.title('highdef')
#    plt.imshow(data['gray'], cmap='gray', interpolation='none')
#    plt.subplot(2, 3, 5)
#    plt.imshow(data['mask'], cmap='gray', interpolation='none')
#    plt.subplot(2, 3, 6)
#    plt.imshow(data['bw'], cmap='gray', interpolation='none')
#    plt.show()
    


#    from pylab import cm
#    from quill.quill import StrokeWidthTransformQuill
#    p = u'/home/fredrik/Dropbox/htr-data/sdhk till Anders'
#    for data in sdhk.imageGenerator([1826, 33780, 1002, 25299, 26063, 21400]):
#        cv2.imwrite(os.path.join(p, str(data['id']) + '.png'), data['gray'])
#        H = StrokeWidthTransformQuill(data['gray'], mask=data['mask']).generate()
#        H /= np.max(H)
#        I = np.asarray(np.round(cm.jet(H)[:, :, :-1]*255), dtype=np.uint8)
#        cv2.imwrite(os.path.join(p, "quill_" + str(data['id']) + ".png"), cv2.cvtColor(I, cv2.COLOR_RGB2BGR))
