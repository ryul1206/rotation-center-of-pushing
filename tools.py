import numpy as np


def Tmat2D(rad, x_trans, y_trans):
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array([[c, -s, x_trans], [s, c, y_trans], [0, 0, 1]])


def TmatDot(Tmat, xys):
    is_single = len(xys.shape) == 1
    if is_single:
        xys = xys.reshape(1, -1)
    _xys = np.vstack((xys.transpose(), np.ones(xys.shape[0])))
    res = np.dot(Tmat, _xys)[:-1].transpose()
    return res[0] if is_single else res


def polygon_area(xs, ys):
    # https://stackoverflow.com/a/30408825/7128154
    return 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def polygon_centroid(xy_points):
    # https://stackoverflow.com/a/66801704
    # xy = np.array([xs, ys])
    xy = xy_points.transpose()
    xs = xy[0, :]
    ys = xy[1, :]
    c = np.dot(
        xy + np.roll(xy, 1, axis=1), xs * np.roll(ys, 1) - np.roll(xs, 1) * ys
    ) / (6 * polygon_area(xs, ys))
    return c


if __name__ == "__main__":
    xs = [0, 0, 2]
    ys = [1, -1, 0]
    xy_points = np.array([xs, ys]).transpose()
    print(polygon_centroid(xy_points), "expect: [2/3, 0]")
