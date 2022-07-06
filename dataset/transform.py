import numpy as np
import cv2
def get_transform(center, scale, res, rot=0):
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        a = float(res[1]) / h
        b = float(res[0]) / h
        rot_rad = rot * np.pi / 180
        rot_matrix = np.zeros((3, 3))
        rot_matrix[0, 0] = a * np.cos(rot_rad)
        rot_matrix[0, 1] = b * np.sin(rot_rad)
        rot_matrix[0, 2] = -a * np.cos(rot_rad) * float(center[0]) - b * np.sin(rot_rad) * float(center[1]) + res[0] / 2
        rot_matrix[1, 0] = -b * np.sin(rot_rad)
        rot_matrix[1, 1] = a * np.cos(rot_rad)
        rot_matrix[1, 2] = b * np.sin(rot_rad) * float(center[0]) - a * np.cos(rot_rad) * float(center[1]) + res[1] / 2
        rot_matrix[2, 2] = 1
        t=rot_matrix
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)

def crop(img, center, scale, res, rot=0):
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]

    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    return cv2.resize(new_img, res)

def inv_mat(mat):
    ans = np.linalg.pinv(np.array(mat).tolist() + [[0,0,1]])
    return ans[:2]

def kpt_affine(kpt, mat):
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    return np.dot( np.concatenate((kpt, kpt[:, 0:1]*0+1), axis = 1), mat.T ).reshape(shape)

def resize(im, res):
    return np.array([cv2.resize(im[i],res) for i in range(im.shape[0])])