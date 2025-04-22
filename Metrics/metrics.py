import cv2 as cv

from sewar.full_ref import uqi, vifp, scc, sam, ergas, rase
from skimage.metrics import structural_similarity as ssim

    
class UQI:
    def process(img1, img2):
        return uqi(img1, img2)
    
class VIFP:
    def process(img1,img2):
        return vifp(img1,img2)
    
class SCC:
    def process(img1, img2):
        return scc(img1,img2)
    
class SAM:
    def process(img1,img2):
        return sam(img1, img2)
    
class ERGAS:
    def process(img1,img2):
        return ergas(img1,img2)
    
class RASE:
    def process(img1,img2):
        return rase(img1,img2)
    
class SIFT:
    def process(img1,img2):
        
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test
        good = []
        for a,b in matches:
            if a.distance < 0.65*b.distance:
                good.append([a])
        return len(good)
    
class ORB:
    def process(img1,img2):
        
        orb = cv.ORB_create()
        
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors.
        matches = bf.match(des1,des2)
        
        return len(matches)
    
class SSIM:
    def process(img1,img2):
        (score, diff) = ssim(img1, img2, full=True)
        return score


