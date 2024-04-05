# import the necessary packages
from operator import itemgetter, attrgetter
import numpy as np
import cv2, math, argparse, signal, time
from . import sba4pm, getcentroid
from shapely.geometry import Polygon, LineString, MultiLineString
import importlib
importlib.reload(sba4pm)

def pointLocationInContour(point, contour):
    for count,value in enumerate(contour):
        compare = (value == point)
        compare = compare[0, 0] and compare[0,1]
        if compare == True:
            break
    return count

def partialContourBetween2Points(A,B,contour):
    counterA = pointLocationInContour(A,contour)
    counterB = pointLocationInContour(B,contour)
    if counterA < counterB: 
        segment = contour[counterA:(counterB+1)]
    else:
        segment = np.concatenate((contour,contour), axis = 0)[counterA:(len(contour)+counterB)]
    return segment

def point2set(p1,p2):
    set = []
    p = p1
    d = p2-p1
    N = np.max(np.abs(d))
    if (p1 == p2).all():
        s = 0
    else:
        s = d/N
    set.append(np.rint(p).astype('int'))
    for ii in range(0,N):
        p = p+s;
        set.append(np.rint(p).astype('int'))
    set = np.array(set)
    return set

def contourMerge(contourA, contourB):
    contourB = contourB[::-1]
    contourAtoB = point2set(contourA[-1],contourB[0])
    contourBtoA = point2set(contourB[-1],contourA[0])
    contour = np.concatenate((contourA, contourAtoB[1:], contourB, contourBtoA[1:-1]), axis = 0)
    return contour

def getCentroid(contour):
    try:
        M = cv2.moments(contour)
        cx, cy = getcentroid.get_center_of_half_area_line(contour)
    except:
        cx, cy = 0, 0
    return np.array([int(cx), int(cy)])

def midPointCalibrate(a1,m1,p1,p2,m2,a2):
    simple_shape = Polygon([a1,m1,p1,p2,m2,a2])
    y_min = min(simple_shape.exterior.xy[1])
    y_max = max(simple_shape.exterior.xy[1])
    m1c, m2c = (a1+p1)/2,(a2+p2)/2
    if (m1c[0] - m2c[0]) != 0:
        a = (m1c[1] - m2c[1]) / (m1c[0] - m2c[0])
        b = m1c[1] - a * m1c[0]
        m1p = [(y_min-b)/a,y_min]
        m2p = [(y_max-b)/a,y_max]
    else:
        m1p = [m1c[0],y_min]
        m2p = [m1c[0],y_max]
    mid_line = LineString([m1p,m2p])
    m1_update,m2_update = mid_line.intersection(simple_shape).coords
    return np.asarray(m1_update).astype(int), np.asarray(m2_update).astype(int)

def imageWithMasks(im1, im2, im3, c, a):
    im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
    mask = 255 - np.stack((np.asarray(cv2.bitwise_or(im2, im3)).astype(np.uint8),)*3, axis=-1)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    mask = np.uint8(mask)
    colr = np.zeros(im1.shape, np.uint8)
    colr[:] = c
    res1 = np.where(mask==(255,255,255), im1, (0,0,0))
    res2 = cv2.addWeighted(im1, a, colr, 1 - a, 0, colr)
    res2 = np.where(mask==(0,0,0), res2, (0,0,0))
    res3 = cv2.bitwise_or(res1,res2).astype(np.uint8)
    return res3

class Vertebra():
    def getSurfaceContours(self):
        points = [self.A.a1, self.A.a2, self.A.p2, self.A.p1]
        contour = self.A.contour
        topSegmentA = partialContourBetween2Points(points[3],points[0],contour)
        bottomSegmentA = partialContourBetween2Points(points[1],points[2],contour)    
        points = [self.B.a1, self.B.a2, self.B.p2, self.B.p1]
        contour = self.B.contour
        topSegmentB = partialContourBetween2Points(points[3],points[0],contour)
        bottomSegmentB = partialContourBetween2Points(points[1],points[2],contour)
        topSurfaceContour = contourMerge(topSegmentA, topSegmentB) 
        bottomSurfaceContour = contourMerge(bottomSegmentA, bottomSegmentB)
        return topSegmentA, topSegmentB, topSurfaceContour, bottomSegmentA, bottomSegmentB, bottomSurfaceContour

    def getSurfaceCentroids(self):
        topSurfaceCentroid = getCentroid(self.topSurfaceContour)
        bottomSurfaceCentroid = getCentroid(self.bottomSurfaceContour)
        if (topSurfaceCentroid == [0, 0]).all():
            topSurfaceCentroid = [bottomSurfaceCentroid[0], 0]
            self.log = np.append(self.log,"topSurfaceCentroid was modified based on bottomSurfaceCentroid")
        elif (bottomSurfaceCentroid == [0, 0]).all():
            bottomSurfaceCentroid = [topSurfaceCentroid[0], image.shape[1]]
            self.log = np.append(self.log,"bottomSurfaceCentroid was modified based on topSurfaceCentroid")
        elif (topSurfaceCentroid == bottomSurfaceCentroid).all():
            self.log = np.append(self.log,"topSurfaceCentroid was at bottomSurfaceCentroid")
        return topSurfaceCentroid, bottomSurfaceCentroid

    def getPoints(self):
        a1 = np.asarray((self.A.a1 + self.B.a1)/2).astype(int)
        a2 = np.asarray((self.A.a2 + self.B.a2)/2).astype(int)
        p1 = np.asarray((self.A.p1 + self.B.p1)/2).astype(int)
        p2 = np.asarray((self.A.p2 + self.B.p2)/2).astype(int)
        m1 = self.topSurfaceCentroid
        m2 = self.bottomSurfaceCentroid
        try:
            m1, m2 = midPointCalibrate(a1,m1,p1,p2,m2,a2)
        except:
            m1, m2 = self.topSurfaceCentroid, self.bottomSurfaceCentroid
        centroid = np.asarray((self.A.centroid + self.B.centroid)/2).astype(int)
        return a1, a2, m1, m2, p1, p2, centroid
    
    def getHeights(self):
        ha = sba4pm.euclidean2D(self.a1, self.a2)
        hm = sba4pm.euclidean2D(self.m1, self.m2)
        hp = sba4pm.euclidean2D(self.p1, self.p2)
        return ha, hm, hp
    
    def getLoss(self):
        loss = 1 - (np.min(np.asarray([self.ha, self.hm, self.hp])) / np.max(np.asarray([self.ha, self.hm, self.hp])))
        return loss
    
    def getGrade(self):
        if self.l < 0.2: GRADE = 0
        elif self.l >= 0.2 and self.l < 0.25: GRADE = 1
        elif self.l >= 0.25 and self.l < 0.40: GRADE = 2
        elif self.l >= 0.40: GRADE = 3
        return GRADE
    
    def getShape(self):
        heights = [self.ha, self.hm, self.hp]
        return heights.index(min(heights))
    
    def __init__(self, image, masks):
        self.I = image
        self.A = sba4pm.SBA(masks[0])
        self.B = sba4pm.SBA(masks[1])
        self.valid = 1
        self.log = []
        if self.A.valid == 1 and self.B.valid == 1:
            self.topSegmentA, self.topSegmentB, self.topSurfaceContour, self.bottomSegmentA, self.bottomSegmentB, self.bottomSurfaceContour = self.getSurfaceContours()
            self.topSurfaceCentroid, self.bottomSurfaceCentroid = self.getSurfaceCentroids()
            self.a1, self.a2, self.m1, self.m2, self.p1, self.p2, self.centroid = self.getPoints()
            self.ha, self.hm, self.hp = self.getHeights()
            self.l = self.getLoss()
            self.g = self.getGrade()
            self.s = self.getShape()
            self.log = np.append(self.log,"Vertebra created")
        elif self.A.valid == 0 and self.B.valid == 0:
            self.valid, self.log = 0, np.append(self.log,"Both mask failed")
        elif self.A.valid == 1 and self.B.valid == 0:
            self.valid, self.log = 0, np.append(self.log,"Mask B failed")
        elif self.A.valid == 0 and self.B.valid == 1:
            self.valid, self.log = 0, np.append(self.log,"Mask A failed")  