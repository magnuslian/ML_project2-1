import numpy as np

#Some filters which can be applied to the predicted black/white image for cleaning

def islandfilter(image):
    padimage = np.lib.pad(image, ((2, 2), (2, 2)), 'reflect')
    filter1 = np.array(((1, 1, 1, 0), (1, 0, 1, 0), (1, 1, 1, 0), (0, 0, 0, 0)))
    filter21 = np.array(((1, 0, 1, 0), (0, 0, 0, 0), (1, 0, 1, 0), (0, 0, 0, 0)))
    filter22 = np.array(((0, 1, 0, 0), (1, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 0)))
    filter3 = np.array(((1, 1, 1, 1), (1, 0, 0, 1), (1, 1, 1, 1), (0, 0, 0, 0)))
    filter4 = np.array(((1, 1, 1, 0), (1, 0, 1, 0), (1, 0, 1, 0), (1, 1, 1, 0)))
    filter5 = np.array(((1, 1, 1, 1), (1, 0, 0, 1), (1, 0, 0, 1), (1, 1, 1, 1)))
    for ih in range (image.shape[0]):
        for iw in range (image.shape[1]):
            subim = padimage[ih+1:ih+5, iw+1:iw+5]
            if image[ih,iw] == 1:
                if np.sum(subim*filter1)==0:
                    image[ih,iw] = 0
                if np.sum(subim*filter21)==1 and np.sum(subim*filter22)==0:
                    image[ih,iw] = 0
                if np.sum(subim*filter3)==0:
                    image[ih,iw] = 0
                if np.sum(subim*filter4)==0:
                    image[ih,iw] = 0
                if np.sum(subim*filter5)==0:
                    image[ih,iw] = 0
            if image[ih,iw] == 0:
                if np.sum(subim*filter1)==8:
                    image[ih,iw] = 1
    return image

def directionfilterver(image):
    ver = np.std(np.mean(image, axis=0))
    hor = np.std(np.mean(image, axis=1))
    print ('ver',ver)
    print ('hor', hor)
    return (ver >= hor, ver-hor >= 0.15)

def rectfilterver(image):
    new = np.ones((12))
    new2 = np.ones((20))
    for iw in range (image.shape[1]):
        for ih in (0,8,17,24):
            part = image[ih:ih+14,iw]
            if np.sum(part)>11:
                image[ih+1:ih+13,iw]=new
        for ih2 in (0,16):
            part = image[ih:ih +22, iw]
            if np.sum(part) > 17:
                image[ih + 1:ih + 21, iw] = new2
    return image
def rectfilterhor(image):
    new = np.ones((1,12))
    new2 = np.ones((1, 20))
    for ih in range (image.shape[0]):
        for iw in (0,8,17,24):
            part = image[ih,iw:iw+14]
            if np.sum(part)>12:
                image[ih,iw+1:iw+13]=new
        for iw2 in (0,16):
            part = image[ih,iw:iw2 +22]
            if np.sum(part) > 19:
                image[ih,iw+1:iw+21] = new2
    return image

def filterver(image):
    w = 8
    backg = 4
    ratio = 0.51
    for iw in range (image.shape[0]):
        isroad = False
        for ih in range (image.shape[1]-(backg)):
            ground = image[iw:iw + w, ih:ih + backg]
            road = image[iw:iw + w, ih + backg:ih + backg +1]
            mg1 = np.mean(ground)
            mroad = np.mean(road)
            if not isroad:
                if mg1 < ratio and mroad == 1.0:
                    # Change background1 to 0
                    for n2pa in range(ground.shape[1]):
                        for n1pa in range(ground.shape[0]):
                            pa = ground[n1pa, n2pa]
                            if pa == 1 and np.sum(ground[n1pa,:]) != ground.shape[1]:
                                ground[n1pa, n2pa] = 0
                isroad = True
            else:
                if mroad != 1.0:
                    isroad = False
            image[iw:iw + w, ih:ih + backg] = ground
            image[iw:iw + w, ih + backg:ih + backg +1] = road
    return image

def filterverd(image):
    w = 8
    backg = 4
    ratio = 0.51
    for ih in range (image.shape[0]):
        isroad = False
        for iw in range (image.shape[1]-(backg),0,-1):
            ground = image[ih:ih + w, iw-backg:iw]
            road = image[ih:ih + w, iw-backg-1:iw-backg]
            mg1 = np.mean(ground)
            mroad = np.mean(road)
            if not isroad:
                if mg1 < ratio and mroad == 1.0:
                    # Change background1 to 0
                    for n2pa in range(ground.shape[1]):
                        for n1pa in range(ground.shape[0]):
                            pa = ground[n1pa, n2pa]
                            if pa == 1 and np.sum(ground[n1pa,:]) != ground.shape[1]:
                                ground[n1pa, n2pa] = 0
                isroad = True
            else:
                if mroad != 1.0:
                    isroad = False
            image[ih:ih + w, iw-backg:iw] = ground
            image[ih:ih + w, iw-backg-1:iw-backg] = road
    return image

def filterhor(image):
    w = 8
    backg = 4
    ratio = 0.51
    for ih in range (image.shape[1]):
        isroad = False
        for iw in range (image.shape[0]-(backg)):
            ground = image[iw:iw + backg,ih:ih + w]
            road = image[ iw + backg:iw +backg+1,ih:ih + w]
            mg1 = np.mean(ground)
            mroad = np.mean(road)
            if not isroad:
                if mg1 < ratio and mroad == 1.0:
                    # Change background1 to 0
                    for n2pa in range(ground.shape[0]):
                        for n1pa in range(ground.shape[1]):
                            pa = ground[n2pa, n1pa]
                            if pa == 1 and np.sum(ground[:,n1pa]) != ground.shape[0]:
                                ground[n2pa, n1pa] = 0
                isroad = True
            else:
                if mroad != 1.0:
                    isroad = False
            image[iw:iw + backg,ih:ih + w] = ground
            image[ iw + backg:iw + backg+1,ih:ih + w] = road
    return image

def filterhord(image):
    w = 8
    backg = 4
    ratio = 0.51
    for ih in range (image.shape[1]):
        isroad = False
        for iw in range (image.shape[0]-(backg),0,-1):
            ground = image[iw-backg:iw,ih:ih + w]
            road = image[ iw-backg-1:iw-backg,ih:ih + w]
            mg1 = np.mean(ground)
            mroad = np.mean(road)
            if not isroad:
                if mg1 < ratio and mroad == 1.0:
                    # Change background1 to 0
                    for n2pa in range(ground.shape[0]):
                        for n1pa in range(ground.shape[1]):
                            pa = ground[n2pa, n1pa]
                            if pa == 1 and np.sum(ground[:,n1pa]) != ground.shape[0]:
                                ground[n2pa, n1pa] = 0
                isroad = True
            else:
                if mroad != 1.0:
                    isroad = False
            image[iw-backg:iw,ih:ih + w] = ground
            image[iw-backg-1:iw-backg,ih:ih + w] = road
    return image

def bigroad(image):
    padimage = np.lib.pad(image, ((1, 1), (1, 1)), 'constant', constant_values=((0, 0), (0, 0)))
    hor = np.sum(padimage, axis=0)
    ver = np.sum(padimage, axis=1)
    allv = np.ones((1,36))
    allh = np.ones(36)
    for i in range (38):
        if ver[i]>34 and ver[i-1]>30 and ver[i-1]<ver[i]:
            padimage[i,2:-2]=allv
            padimage[i - 1]=padimage[i]
        if ver[i] > 34 and ver[i+1] > 30 and ver[i+1]<ver[i]:
            padimage[i, 2:-2] = allv
            padimage[i + 1] = padimage[i]
        if hor[i]>34 and hor[i-1]>30 and hor[i-1]<hor[i]:
            padimage[2:-2,i] = allh
            padimage[:,i - 1]= padimage[:,i]
        if hor[i]>34 and hor[i+1]>30 and hor[i+1]<hor[i]:
            padimage[2:-2,i] = allh
            padimage[:,i + 1]=padimage[:,i]
    return padimage[1:-1,1:-1]

def filterh(images, ratioroad=0.79, ratiobackground=0.31):
    for nimage in range(images.shape[0]):
        image = images[nimage]

        image = rectfilterhor(image)
        image = rectfilterver(image)

        image = islandfilter(image)
        image = islandfilter(image)
        #close_img = ndimage.binary_closing(image)

        image = filterver(image)
        image = filterverd(image)
        image = filterhor(image)
        image = filterhord(image)

        image = islandfilter(image)

        image = rectfilterhor(image)
        image = rectfilterver(image)
        image = bigroad(image)

        """
        dir=directionfilterver(image)
        if dir[0] and dir[1] :
            image = verticalfilter(image)
        elif dir[1]:
            image = horitzontalfilter(image)
        """
        images[nimage] = image

    return images