#!/usr/bin/env python


# Python 2/3 compatibility
from __future__ import print_function
from pprint import pprint
import numpy as np
import cv2
from extract import seg_img

min_kp1_size = 10
min_kp2_size = 10
kp_divisor = 200*200
good_color = (0,255,0)
bad_color = (0,0,0)
ratio = 0.7

def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.xfeatures2d.SIFT_create(edgeThreshold=5)
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.xfeatures2d.SURF_create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(nfeatures=1000, WTA_K=3, scoreType=cv2.ORB_FAST_SCORE)
        norm = cv2.NORM_HAMMING2
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, min_kp1_size, kp2, min_kp2_size, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            n = m[0]
            if kp1[n.queryIdx].size > min_kp1_size and kp2[n.trainIdx].size > min_kp2_size:
                mkp1.append( kp1[n.queryIdx] )
                mkp2.append( kp2[n.trainIdx] )
        
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    
    return p1, p2

def match_and_draw(matcher, kp1, desc1, min_kp1_size, kp2, desc2, min_kp2_size, img1, img2, si, di):
    raw_matches = matcher.knnMatch(
        desc1, trainDescriptors=desc2, k=2)  # 2
    p1, p2 = filter_matches(kp1, min_kp1_size, kp2, min_kp2_size, raw_matches, ratio)
    good = []
    for m, n in raw_matches:
        if m.distance < ratio*n.distance:
            good.append([m])
    
    if len(p1) >= 5:
        h, status = cv2.findHomography(p1, p2, cv2.RANSAC, 10.0)
        inliner = []
        for i in range(status.size):
            if status[i] > 0:
                inliner.append(good[i])
        
        matched = np.sum(status) / len(status) * 100
        matches = np.sum(status)           
        _r = 1.0
        non_linear = False        
        try:
            img_warp = cv2.warpPerspective(img1, h, (img2.shape[1],img2.shape[0]))
            if img_warp is not None and len(img_warp) > 0:
                img2_clone = img2
                last_r = 100
                for i in range(6):
                    img_warp = cv2.pyrDown(img_warp)
                    img2_clone = cv2.pyrDown(img2_clone)
                    if min(img_warp.shape[:2]) < 20:
                        break;
                    img2_bw = cv2.threshold(img2_clone, 127, 255, cv2.THRESH_BINARY)[1]
                    img_warp_bw = cv2.threshold(img_warp, 127, 255, cv2.THRESH_BINARY)[1]

                    mask_inv = cv2.bitwise_not(img2_bw)
                    xor = cv2.bitwise_and(img_warp_bw, mask_inv)
                    img3 = cv2.hconcat([img_warp, img2_clone, mask_inv, xor])
                    _rows, _cols = xor.shape[:2]
                    _size = _rows * _cols
                    _diff = np.count_nonzero(xor)
                    _orig_size = np.count_nonzero(img2_clone)
                    _r = float( _diff / _orig_size) # _diff / _size
                    if _r > last_r:
                        print("non linearity detected: " + str(last_r) + " vs " + str(r))
                        non_linear = True
                    last_r = _r
                    cv2.imwrite("warped-" + str(int(matched)) + "-" + str(int(_r * 100)) + "-scale-" + str(i) + ".jpg", img3)
                
        except:
            img_warp = None
        return matched, matches, inliner, _r, non_linear
    else:
        return 0, 0, good, img1, 1.0, non_linear

def compare_two(feature_name, fn1, fn2):
    src = cv2.imread(fn1)
    dst = cv2.imread(fn2)

    detector, matcher = init_feature(feature_name)
    
    if src is None:
        print('Failed to load target image:', fn1)
        sys.exit(1)

    if dst is None:
        print('Failed to load truth image:', fn2)
        sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)
    
    srcImg, srcLoc = seg_img(src, fn1)
    dstImg, dstLoc = seg_img(dst, fn2)
    print("total sub images:" + str(len(srcLoc)) + " " + str(len(dstLoc)))
    print('using', feature_name)
    si = 0
    draws = []
    for i in srcLoc:
        img1 = srcImg[i[1]:i[1] + i[3], i[0]:i[0] + i[2]] 
        if i[3] < 100 or i[2] < 100:
            print("src too small")
            continue
        src_rows, src_cols = img1.shape[:2]
        src_size = src_rows * src_cols
        src_r = float(np.count_nonzero(img1) / src_size)
        kp1, desc1 = detector.detectAndCompute(img1, None)
        min_kp1_size = int((i[3]*i[2])/kp_divisor)
        goodKp1 = 0
        for ki in range (len(kp1)):
            if kp1[ki].size > min_kp1_size:
                goodKp1 = goodKp1 + 1
        if goodKp1 == 0:
            continue
        best_matches = 0
        best_matched = 0
        best_pt1 = (0,0)
        best_pt2 = (0,0)
        best_max_matched_ratio = 0
        best_min_matched_ratio = 0
        best_index = 0
        best_residual = 0
        di = 0
        for j in dstLoc:
            if j[3] < 100 or j[2] < 100:
                print("dst too small")
                continue
            img2 = dstImg[j[1]:j[1] + j[3], j[0]:j[0] + j[2]]               
            dst_rows, dst_cols = img2.shape[:2]
            img2_resize = cv2.resize(img2.copy(), (src_cols, src_rows))
            dst_r = float(np.count_nonzero(img2_resize) / src_size)
            diff_r = src_r / dst_r
            print("src: " + str(img1.shape[0]) + " " + str(img1.shape[1]) + " " + str(src_r))
            print("dst: " + str(img2_resize.shape[0]) + " " + str(img2_resize.shape[1])  + " " + str(dst_r))
            if diff_r > 1.2 or diff_r < 0.8:
                print("size diff")
                continue

            kp2, desc2 = detector.detectAndCompute(img2_resize, None)
            goodKp2 = 0
            min_kp2_size = int((img2_resize.shape[0]*img2_resize.shape[1])/kp_divisor)
            for ki in range (len(kp2)):
                if kp2[ki].size > min_kp2_size:
                    goodKp2 = goodKp2 + 1
            if goodKp2 == 0:
                print("no good kp")
                continue
            try:
                matched, matches, good, residual, non_linear = match_and_draw(matcher, kp1, desc1, min_kp1_size, kp2, desc2, min_kp2_size, img1, img2_resize, si, di)
            except:
                matched = 0
                matches = 0
                good = []
                raw = []
                residual = 1.0
                non_linear = True
            print("src %d big kp %d features %d, dst %d big kp %d features %d non_linear %s" % (si, goodKp1, len(desc1), di, goodKp2, len(desc2), non_linear))
            min_features = goodKp2 #len(desc1) #(len(desc1) + len(desc2))/2
            max_features = len(desc2)
            min_matched_ratio = matches / min_features
            max_matched_ratio = matches / max_features
            if best_matched == 0:
                best_matched = matched
                best_matches = matches
                best_max_matched_ratio = max_matched_ratio
                best_pt1 = (i[0], i[1])
                best_pt2 = (i[0] + i[2], i[1] + i[3])
                best_min_matched_ratio = min_matched_ratio
                best_residual = residual
            else:
                pick_it = False
                if matched > 80 and max_matched_ratio > 0.5:  
                    if matched > best_matched:
                        pick_it = True
                if matched > 70 and best_min_matched_ratio > 0.01:
                    if min_matched_ratio >= best_min_matched_ratio:
                        pick_it = True
                if pick_it:
                    best_max_matched_ratio = max(max_matched_ratio, best_max_matched_ratio)
                    best_pt1 = (i[0], i[1])
                    best_pt2 = (i[0] + i[2], i[1] + i[3])
                    best_min_matched_ratio = max(min_matched_ratio, best_min_matched_ratio)
                    best_matched = max(matched, best_matched)
                    best_matches = max(matches, best_matches)
                    best_residual = min(best_residual, residual)
            di = di + 1

        # just need to tune matched and matched_raio
        if best_matched > 80 and best_max_matched_ratio > 0.5:    
            print('%d%% matched, good matches %s, matched_ratio %s residual %s' % (best_matched, best_matches, best_max_matched_ratio, best_residual))
            if fn1 != fn2 :
                srcClone = src.copy()      
                draws.append((best_pt1, best_pt2, good_color))
                cv2.rectangle(srcClone, best_pt1, best_pt2, good_color, 10)
                cv2.imwrite("good-" + str(int(best_matched)) + "-" + str(si) + "-" + str(best_index) + ".jpg", srcClone)
            else:
                srcClone = src.copy()      
                cv2.rectangle(srcClone, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), good_color, 10)
                cv2.rectangle(srcClone, best_pt1, best_pt2, good_color, 10)
                cv2.imwrite("good-" + str(int(best_matched))+ "-" + str(si) + "-" + str(best_index) + ".jpg", srcClone)
                draws.append((best_pt1, best_pt2, good_color))
        else:
            if best_residual < 0.25:
                print('%d%% matched, good matches %s matched_ratio %s residual %s' % (best_matched, best_matches, best_min_matched_ratio, best_residual))
                draw_color = bad_color
                prefix = "failed-"
                if (best_matched > 80 and best_min_matched_ratio > 0.1) or (best_matched > 90 and best_matches >= 10) or (best_matched > 75 and best_min_matched_ratio > 0.075 and best_matches > 20):
                    draw_color = good_color
                    prefix = "matched-"
                draws.append((best_pt1, best_pt2, draw_color))
                if fn1 != fn2 :
                    srcClone = src.copy()      
                    cv2.rectangle(srcClone, best_pt1, best_pt2, draw_color, 10)
                    cv2.imwrite(prefix + str(int(best_matched))+ "-" +  str(int(best_min_matched_ratio*100)) + "-" + str(int(best_residual*100)) + "-" + str(si) + "-" + str(best_index) + ".jpg", srcClone)
                else:                  
                    srcClone = src.copy()          
                    cv2.rectangle(srcClone, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), draw_color, 10)
                    cv2.rectangle(srcClone, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), draw_color, 10)
                    cv2.imwrite(prefix + str(int(matched))+ "-" +  str(int(min_matched_ratio*100)) + "-" + str(int(best_residual*100)) + "-" + str(si) + "-" + str(di) + ".jpg", srcClone)

        si = si + 1
    return draws

if __name__ == '__main__':
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'akaze')
    # fn1 is the check target, fn2 is known real one
    fn1, fn2 = args
    draws = compare_two(feature_name, fn1, fn2)
    srcClone = cv2.imread(fn1)
    for i in draws:
        pt1, pt2, color = i[0],i[1],i[2]
        cv2.rectangle(srcClone, pt1, pt2, color, 10)
    cv2.imwrite("result-" + fn1, srcClone)  
