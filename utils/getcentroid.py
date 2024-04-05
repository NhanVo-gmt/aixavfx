import numpy as np
import cv2
from shapely.geometry import Polygon, LineString, MultiLineString, Point, MultiPoint, GeometryCollection
from skimage.morphology import skeletonize, medial_axis
from scipy.optimize import minimize_scalar
from scipy.ndimage.morphology import distance_transform_edt

H, W = 500, 500

def get_center_of_mass(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx, cy

def split_mask_by_line(mask, centroid:tuple, theta_degrees:float, eps:float = 1e-4):
    h, w = mask.shape[:2]
    mask = mask[..., None]
    cx, cy = centroid
    # convert theta first to radians and then to line slope(s)
    theta_degrees = np.atleast_1d(theta_degrees)
    theta_degrees = np.clip(theta_degrees, -90+eps, 90-eps)
    theta_rads = np.radians(theta_degrees)
    slopes = np.tan(theta_rads)[:, None]
    # define the line(s)
    x = np.arange(w, dtype="int32")
    y = np.int32(slopes * (x - cx) + cy)
    _y = np.arange(h, dtype="int32")
    # split the input mask into two halves by line(s)
    m = (y[..., None] <= _y).T
    m1 = (m * mask).sum((0,1))
    m2 = ((1 - m) * mask).sum((0,1))
    m2 = m2 + eps if m2==0 else m2
    # calculate the resultant masks ratio
    ratio = m1/m2
    return (x.squeeze(), y.squeeze()), ratio

def get_half_area_line(mask, centroid: tuple, eps: float = 1e-4):
    # find the line that splits the input mask into two equal area halves
    minimize_fun = lambda theta: abs(1. - split_mask_by_line(mask, centroid, theta, eps=eps)[1].item())
    bounds = np.clip((-90, 90), -90 + eps, 90 - eps)
    res = minimize_scalar(minimize_fun, bounds=bounds, method='bounded')
    theta_min = res.x
    line, _ = split_mask_by_line(mask, centroid, theta_min)
    return line

def get_representative_point(cnt):
    poly = Polygon(cnt.squeeze())
    cx = poly.representative_point().x
    cy = poly.representative_point().y
    return cx, cy

def get_skeleton_center_of_mass(cnt):
    mask = draw_contour_on_mask((H,W), cnt)
    skel = medial_axis(mask//255).astype(np.uint8) #<- medial_axis wants binary masks with value 0 and 1
    skel_cnt,_ = cv2.findContours(skel,1,2)
    skel_cnt = skel_cnt[0]
    M = cv2.moments(skel_cnt) 
    if(M["m00"]==0): # this is a line
        cx = int(np.mean(skel_cnt[...,0]))
        cy = int(np.mean(skel_cnt[...,1]))
    else:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    return cx, cy

def get_furthest_point_from_edge(cnt):
    mask = draw_contour_on_mask((H,W), cnt)
    d = distance_transform_edt(mask)
    cy, cx = np.unravel_index(d.argmax(), d.shape)
    return cx, cy

def get_furthest_point_from_edge_cv2(cnt):
    mask = draw_contour_on_mask((H,W), cnt)
    dist_img = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    cy, cx = np.where(dist_img==dist_img.max())
    cx, cy = cx.mean(), cy.mean()  # there are sometimes cases where there are multiple values returned for the visual center
    return cx, cy

def get_center_of_half_area_line(cnt):
    mask = draw_contour_on_mask((H,W), cnt, color=1)
    # get half-area line that passes through centroid
    cx, cy = get_center_of_mass(mask)
    line = get_half_area_line(mask, centroid=(cx, cy))
    line = LineString(np.array(list(zip(line))).T.reshape(-1, 2))
    # find the visual center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > 5]
    polys = [Polygon(c.squeeze(1)) for c in contours if len(c) >= 3]  # `Polygon` must have at least 3 points
    cpoint = Point(cx, cy)
    points = []
    for poly in polys:
        if poly.is_valid == False:
            poly = poly.buffer(0)
        isect = poly.intersection(line)
        if isect.is_empty:
            # skip when intersection is empty: this can happen for masks that consist of multiple disconnected parts
            continue
        if isinstance(isect, (MultiLineString, GeometryCollection)):
            # take the line segment intersecting with `poly` that is closest to the centroid point
            isect = isect.geoms[np.argmin([g.distance(cpoint) for g in isect.geoms])]
        if isinstance(isect, Point):
            # sometimes the intersection can be a singleton point
            points.append(isect)
            continue
        isect = isect.boundary
        if poly.intersects(cpoint):
            points = [isect]
            break
        else:
            points.append(isect)

    if len(points) == 0:
        # multiple reasons for this one:
        # - if len(polys)==0
        # - if len(polys)==1, but for some reason the line does not intersect with polygon
        # - if the above search does not match with any points

        return cx, cy

    points = points[np.argmin([p.distance(cpoint) for p in points])]
    if isinstance(points, Point):
        return np.array(points.xy)
    
    points = [np.array(p.xy).tolist() for p in points.geoms]
    visual_center = np.average(points, (0, 2))
    return visual_center

def draw_contour_on_mask(size, cnt, color:int = 255):
    mask = np.zeros(size, dtype='uint8')
    mask = cv2.drawContours(mask, [cnt], -1, color, -1)
    return mask