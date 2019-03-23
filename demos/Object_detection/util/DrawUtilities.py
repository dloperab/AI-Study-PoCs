import math
import cv2
import numpy as np
from .ColorUtilities import ColorUtilities

class DrawUtilities():

    @staticmethod
    def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
        dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**.5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            p = (x, y)
            pts.append(p)
        if style == 'dotted':
            for p in pts:
                cv2.circle(img, p, thickness, color, -1)
        else:
            s = pts[0]
            e = pts[0]
            i = 0
            for p in pts:
                s = e
                e = p
                if i % 2 == 1:
                    cv2.line(img, s, e, color, thickness)
                i += 1

    @staticmethod
    def drawRect(img, x, y, w, h, offset=3,c=(0,255,0)):
        # corner top left
        factor = 8
        cv2.line(img, (x + offset, y + offset), (x + offset,
                                                 y + int(h / factor) + offset), (0, 255, 0), 2)
        cv2.line(img, (x + offset, y + offset),
                 (x + int(w / factor) + offset, y + offset), (0, 255, 0), 2)
        # corner top right
        cv2.line(img, (x + w - offset, y + offset), (x + w -
                                                     int(w / factor) - offset, y + offset), (0, 255, 0), 2)
        cv2.line(img, (x + w - offset, y + offset), (x + w - offset,
                                                     y + int(h / factor) + offset), (0, 255, 0), 2)
        # bottom top left
        cv2.line(img, (x + offset, y + h - offset), (x + offset,
                                                     y + h - int(h / factor) - offset), (0, 255, 0), 2)
        cv2.line(img, (x + offset, y + h - offset),
                 (x + int(w / factor) + offset, y + h - offset), (0, 255, 0), 2)
        # bottom top right
        cv2.line(img, (x + w - offset, y + h - offset), (x + w - \
                 int(w / factor) - offset, y + h - offset), (0, 255, 0), 2)
        cv2.line(img, (x + w - offset, y + h - offset), (x + w - \
                 offset, y + h - int(h / factor) - offset), (0, 255, 0), 2)

    @classmethod
    def drawCenter(cls, img, x, y, w, h,c=(0,255,0)):
        # draw origin
        origin = (math.floor(x + w / 2), math.floor(y + h / 2))
        cv2.circle(img, origin, 2, c,-1 )
        cls.drawline(
            img,
            (origin[0] - 20,
             origin[1]),
            (origin[0] + 21,
             origin[1]),
            c,
            thickness=1,
            style='dotted',
            gap=5)
        cls.drawline(
            img,
            (origin[0],
             origin[1] - 20),
            (origin[0],
             origin[1] + 21),
            c,
            thickness=1,
            style='dotted',
            gap=5)

    @staticmethod
    def drawGrid(img,x, y, w, h, steps=40, alpha=0.1,c=(0,255,0)):        
        roi = img[y:y+h,x:x+w].copy()
        overlay = np.zeros(roi.shape, dtype=roi.dtype)
        # vertical lines
        vlines = np.arange(0, overlay.shape[0], steps)
        # horizontal lines
        hlines = np.arange(0, overlay.shape[1], steps)

        
        origin = (math.floor(overlay.shape[1] / 2), math.floor(overlay.shape[0] / 2))        
        roih, roiw, roic = roi.shape
        r = math.floor(roih / 2) if roiw > roih else math.floor(roiw)
        cv2.circle(overlay, origin,r, c,-1 )

        # draw lines        
        for i in hlines:
            cv2.line(overlay, (i, 0), (i, overlay.shape[0]), (211, 211, 211), 1)
        for j in vlines:
            cv2.line(overlay, (0, j), (overlay.shape[1], j), (211, 211, 211), 1)

        roi = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)            
        
        img[y:y+h,x:x+w] = roi


        
    @staticmethod
    def drawPath(pts, frame, n=32,c=(0,255,0)):
        #colors_dict= ColorUtilities.rainbow_gradient(n=n)
        color_hex = ColorUtilities.RGB2hex(c)
        colors_dict= ColorUtilities.linear_gradient(color_hex,n=n)
        R = colors_dict["r"]
        G = colors_dict["g"]
        B = colors_dict["b"]
        gradient = list(zip(R,G,B))
        # loop over the set of tracked points
        for i in np.arange(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(n / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], gradient[i], thickness)
