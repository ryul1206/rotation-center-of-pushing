from mpl_toolkits.mplot3d import art3d
import numpy as np

"""
CAUTION:
(X, Y) -> (Y, X)
"""

HALF_PI = np.pi / 2.0


def build_static_elements(ax):
    """3D"""
    # Sphere
    def longitude(alpha, sphere_radius=1.0):
        """Longitude (vertical lines)"""
        dt = np.linspace(0, 2 * np.pi, 100)
        _x = sphere_radius * np.cos(dt)
        z = sphere_radius * np.sin(dt)
        x = _x * np.cos(alpha)
        y = _x * np.sin(alpha)
        return (x, y, z)

    def latitude(beta, sphere_radius=1.0):
        """Latitude (horizontal lines)"""
        dt = np.linspace(0, 2 * np.pi, 100)
        radius = sphere_radius * np.cos(beta)
        x = radius * np.cos(dt)
        y = radius * np.sin(dt)
        z = (sphere_radius * np.sin(beta)) * np.ones(len(dt))
        return (x, y, z)

    def sphere(ax, color, width=1.0):
        def repeat(func, linspace):
            for rad in linspace[1:]:
                aligned = np.allclose(rad % HALF_PI, 0.0)
                gain = 1.0 if aligned else 0.5
                x, y, z = func(rad)
                ax.plot(x, y, z, color=color, alpha=gain, linewidth=width * gain)

        odd = 13  # 15 degrees
        repeat(longitude, np.linspace(0, np.pi, odd))
        repeat(latitude, np.linspace(-HALF_PI, HALF_PI, odd))

    sphere(ax, "gray")

    # XYZ axes
    _axes_length = 0.5
    _axes_width = 1.0
    ax.plot([0, _axes_length], [0, 0], [0, 0], color="red", linewidth=_axes_width)
    ax.plot([0, 0], [0, _axes_length], [0, 0], color="green", linewidth=_axes_width)
    ax.plot([0, 0], [0, 0], [0, _axes_length], color="blue", linewidth=_axes_width)

    # Make axes limits
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = [min(xyzlim[0]), max(xyzlim[1])]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)

    # =========== PLANE ===========
    _axes_length = 0.25
    _axes_width = 2.0
    _a, _b = XYZlim
    _alpha = 0.13
    _color = (0.5, 0.5, 0.5, _alpha)
    # Plane of CCW rotations
    top = (
        (_a, _a, 1),
        (_b, _a, 1),
        (_b, _b, 1),
        (_b, _b, 1),
        (_a, _b, 1),
        (_a, _a, 1),
    )
    plane1 = art3d.Poly3DCollection(top, linewidth=0.0)
    plane1.set_facecolor(_color)
    ax.add_collection3d(plane1)
    ax.plot([0, 0], [0, -_axes_length], [1, 1], color="red", linewidth=_axes_width)
    ax.plot([0, _axes_length], [0, 0], [1, 1], color="green", linewidth=_axes_width)

    # Plane of CW rotations
    bot = (
        (_a, _a, -1),
        (_b, _a, -1),
        (_b, _b, -1),
        (_b, _b, -1),
        (_a, _b, -1),
        (_a, _a, -1),
    )
    plane2 = art3d.Poly3DCollection(bot, linewidth=0.0)
    plane2.set_facecolor(_color)
    ax.add_collection3d(plane2)
    ax.plot([0, 0], [0, _axes_length], [-1, -1], color="red", linewidth=_axes_width)
    ax.plot([0, -_axes_length], [0, 0], [-1, -1], color="green", linewidth=_axes_width)

    # Annotations
    ax.text(1, -1, 1, "CCW ICR", (0, 1, 0))
    ax.text(1, -1, -1, "CW ICR", (0, 1, 0))

    # Label
    # ax.xaxis.set_rotate_label(False)
    # ax.yaxis.set_rotate_label(False)
    # ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(r"$v_x$ [m/s]")
    ax.set_ylabel(r"$v_y$ [m/s]")
    ax.set_zlabel(r"$\omega$ [rad]", rotation=0)


class ConstraintsPatch3D:
    def __init__(self, subplot):
        self.elem_FL = LineConstraintPair3D(subplot, True, "green", linewidth=1.0)
        self.elem_FR = LineConstraintPair3D(subplot, False, "green", linewidth=1.0)
        self.elem_WL = LineConstraintPair3D(subplot, True, "red", linewidth=1.0)
        self.elem_WR = LineConstraintPair3D(subplot, False, "red", linewidth=1.0)

    def update(self, sr):
        """
        sr: An instance of the class stable_region.StableRegion
        """
        # (1) Friction cone condition
        fl1, fl2, fr1, fr2 = sr.stable_constraints_of_friction
        self.elem_FL.update(fl1, fl2, True, False)
        self.elem_FR.update(fr1, fr2, False, True)
        # (2) Non-prehensile condition
        wl1, wl2, wr1, wr2 = sr.stable_constraints_of_wrench
        self.elem_WL.update(wl1, wl2, True, False)
        self.elem_WR.update(wr1, wr2, False, True)


class LineConstraintPair3D:
    def __init__(self, subplot, is_ccw_plane, color, linewidth=1.0):
        self.plus_if_ccw_else_minus = 1.0 if is_ccw_plane else -1.0
        self.minus_if_ccw_else_plus = -1.0 if is_ccw_plane else 1.0
        # xs, ys, zs
        self.line_on_sphere = art3d.Line3D(
            (0, 0), (0, 0), (0, 0), linewidth=linewidth, color=color
        )
        subplot.add_line(self.line_on_sphere)

        self._n = 30
        self._xs = np.zeros(self._n * 3 + 1)
        self._ys = np.zeros(self._n * 3 + 1)
        self._zs = np.zeros(self._n * 3 + 1)
        self._sphere_xs = np.zeros(self._n * 3 + 1)
        self._sphere_ys = np.zeros(self._n * 3 + 1)

    @staticmethod
    def _find_infinite_point(slope_m, direction=True):
        """
        - slope_m: slope of the line (m = -a / b) (ax + by + c = 0)
        - direction: True if the infinite point is on the positive side of the line.
        """
        rad = np.arctan(slope_m)
        if direction:
            return np.array([np.cos(rad), np.sin(rad), 0])
        else:
            return np.array([-np.cos(rad), -np.sin(rad), 0])

    def _plane_xy_to_sphere_xyz(self, x, y):
        k = 1.0 / np.sqrt(x ** 2 + y ** 2 + 1.0)
        _x = x * k
        _y = y * k
        _z = k * self.plus_if_ccw_else_minus
        return np.array([_x, _y, _z])

    def update(self, line1, line2, direction1, direction2):
        """
        direction: True if the infinite point is on the positive side of the line.
        -----------------
        FL (Left-hand ICR)
            fl1: y>= left cone의 -alpha (x=상수꼴이면 queryX<=x)
            fl2: y>= right cone의 alpha + 떨어진점 (x=상수꼴이면 queryX>=x)
        FR (Right-hand ICR)
            fr1: y<= left cone의 -alpha + 떨어진점 (x=상수꼴이면 queryX>=x)
            fr2: y<= right cone의 alpha (x=상수꼴이면 queryX<=x)
        WL (Left-hand ICR)
            wl1: y>= left cone과 centroid 사이의 수직이등분선 (x=상수꼴이면 queryX<=x)
            wl2: y>= `(현재)right cone->centroid`방향, centroid에서 r^2/p 거리 (x=상수꼴이면 queryX>=x)
        WR (Right-hand ICR)
            wr1: y<= `(현재)left cone->centroid`방향, centroid에서 r^2/p 거리 (x=상수꼴이면 queryX>=x)
            wr2: y<= right cone과 centroid 사이의 수직이등분선 (x=상수꼴이면 queryX<=x)
        -----------------
            [a1 b1][x] = [-c1]
            [a2 b2][y]   [-c2]
        Case1: (determinant != 0) One intersection point exists
        Case2: (a == 0 => determinant == 0) No intersection point, Overlap region.
        Case3: (b == 0 => determinant == 0) No intersection point, No overlap
        * Now, (a!=0, b!=0, determinant==0) is not possible.
        """
        a1, b1, c1 = line1.standard_form
        a2, b2, c2 = line2.standard_form
        determinant = a1 * b2 - a2 * b1
        if determinant != 0:
            # One Intersetion point
            plane_x = (b1 * c2 - b2 * c1) / determinant
            plane_y = (a2 * c1 - a1 * c2) / determinant
            # (1st point) Intersection on the sphere
            # (2nd point) Infinite point of line1
            # (3rd point) Infinite point of line2
            itsc_xyz = self._plane_xy_to_sphere_xyz(plane_x, plane_y)
            inf1_xyz = self._find_infinite_point(-a1 / b1, direction1)
            inf2_xyz = self._find_infinite_point(-a2 / b2, direction2)
        elif a1 == 0:
            # Case2 (Overlap region)
            # Two lines are parallel. (b1y + c1 = 0) (b2y + c2 = 0)
            _y1 = -c1 / b1
            _y2 = -c2 / b2
            """
            (l1.greater_than_y), (_y2 > _y1)
                T, T => _y2 (xor F)
                T, F => _y1 (xor T)
                F, T => _y1 (xor T)
                F, F => _y2 (xor F)
            """
            xor = (line1.greater_than_y) ^ (_y2 > _y1)
            itsc_xyz = self._plane_xy_to_sphere_xyz(0, _y1 if xor else _y2)
            inf1_xyz = self._find_infinite_point(0.0, direction1)
            inf2_xyz = self._find_infinite_point(0.0, direction2)
        elif b1 == 0:
            # Case3 (No overlap)
            pass
        else:
            raise ValueError("Invalid case")
        self._make_line_vertices(itsc_xyz, inf1_xyz, inf2_xyz)
        self.line_on_sphere.set_data_3d(
            self._sphere_xs, self._sphere_ys, self._zs
        )

    @staticmethod
    def _get_axis_and_angle(xyz1, xyz2):
        axis = np.cross(xyz1, xyz2)
        if np.allclose(axis, 0.0):
            axis = np.array([0.0, 0.0, -1.0])
        else:
            axis /= np.linalg.norm(axis)
        angle = np.arccos(
            np.dot(xyz1, xyz2) / (np.linalg.norm(xyz1) * np.linalg.norm(xyz2))
        )
        return axis, angle

    @staticmethod
    def _Rmat_from_axis_and_angle(u, rad):
        """
        - u: unit axis
        - rad: angle
        """
        x, y, z = u
        c = np.cos(rad)
        s = np.sin(rad)
        _1c = 1.0 - c
        return np.array(
            [
                [c + (x ** 2) * _1c, x * y * _1c - z * s, x * z * _1c + y * s],
                [y * x * _1c + z * s, c + (y ** 2) * _1c, y * z * _1c - x * s],
                [z * x * _1c - y * s, z * y * _1c + x * s, c + (z ** 2) * _1c],
            ]
        )

    def _put_xyz(self, index, xyz):
        self._xs[index] = xyz[0]
        self._ys[index] = xyz[1]
        self._zs[index] = xyz[2]
        return index + 1

    def _interpolation(self, xyz1, xyz2, start_index):
        """
        - xyz1, xyz2: 3D coordinates
        """
        axis, rad = self._get_axis_and_angle(xyz1, xyz2)
        Rmat = self._Rmat_from_axis_and_angle(axis, rad / self._n)
        i = start_index
        p = xyz1
        for _ in range(self._n - 1):
            p = np.dot(Rmat, p)
            self._xs[i] = p[0]
            self._ys[i] = p[1]
            self._zs[i] = p[2]
            i += 1
        return i

    def _make_line_vertices(self, intersection, infp1, infp2):
        # infp1
        # infp1 -> intersection
        next_index = self._put_xyz(0, infp1)
        next_index = self._interpolation(infp1, intersection, next_index)
        # intersection
        # intersection -> infp2
        next_index = self._put_xyz(next_index, intersection)
        next_index = self._interpolation(intersection, infp2, next_index)
        # infp2
        # infp2 -> infp1
        next_index = self._put_xyz(next_index, infp2)
        next_index = self._interpolation(infp2, infp1, next_index)
        # infp1
        self._put_xyz(next_index, infp1)
        """
        CCW plane: x = _y,  y = -_x
        CW plane: x = -_y, y = _x
        """
        self._sphere_xs = self._ys * self.plus_if_ccw_else_minus
        self._sphere_ys = self._xs * self.minus_if_ccw_else_plus


class MouseLocationPatch:
    def __init__(self, subplot):
        dot_size = 0.015
        samples = np.linspace(0, 2 * np.pi, 10)
        self.dot_verts = np.array(  # [[xs], [ys], [zs]]
            [
                np.cos(samples) * dot_size,
                np.sin(samples) * dot_size,
                np.zeros(len(samples)),
            ]
        )
        self.cursor_dot = art3d.Line3D(*self.dot_verts, linewidth=1.5)
        self.cursor_dot_on_sphere = art3d.Line3D(*self.dot_verts, linewidth=1.5)
        subplot.add_line(self.cursor_dot)
        subplot.add_line(self.cursor_dot_on_sphere)

        self.cursor_line = art3d.Line3D(
            (0, 0), (0, 0), (0, 0), linewidth=1.0, color="black"
        )  # xs, ys, zs
        subplot.add_line(self.cursor_line)

    def update(self, event_xy, local_centroid, is_stable):
        """
        y_from_centroid > 0: Left-hand side (CCW, z-axis = +1)
        y_from_centroid < 0: Right-hand side (CW, z-axis = -1)
        """
        # Position of the dot
        # dxyz is on the pusher fame! Not on the slider centroid!
        local_x, local_y = (event_xy[1], event_xy[0])
        is_top = local_y > 0
        dxyz = np.array(
            ([local_y], [-local_x], [1]) if is_top else ([-local_y], [local_x], [-1])
        )
        point_on_plane = self.dot_verts + dxyz
        self.cursor_dot.set_data_3d(*point_on_plane)
        # Position of the dot on the sphere
        """  x^2 + y^2 + z^2 = 1
        x = dxyz[0] * k
        y = dxyz[1] * k
        z = dxyz[2] * k
        ----
        k^2 * (dxyz[0]^2 + dxyz[1]^2 + dxyz[2]^2) = 1
        dxyz[2] == 1
        """
        # np.sqrt(dxyz[0][0] ** 2 + dxyz[1][0] ** 2 + dxyz[2][0] ** 2) >= 1.0
        k = 1.0 / np.sqrt(dxyz[0][0] ** 2 + dxyz[1][0] ** 2 + 1.0)
        xyz_on_sphere = dxyz * k
        """
        origin_P = origin_target_T *  target_P
        origin_target_T = T[[Rz(azimuth)*Ry(HALF_PI - elevation), xyz_on_sphere], [0, 0, 0, 1]]
        Rz(a) = [
            [cos(a), -sin(a), 0],
            [sin(a), cos(a), 0],
            [0, 0, 1]
        ]
        Ry(b) = [
            [cos(b), 0, sin(b)],
            [0, 1, 0],
            [-sin(b), 0, cos(b)]
        ]
        Rz(a)*Ry(b) = [
            [cos(a)*cos(b), -sin(a), cos(a)*sin(b)],
            [sin(a)*cos(b), cos(a), sin(a)*sin(b)],
            [-sin(b), 0, cos(b)]
        ]
        """
        azimuth = np.arctan2(xyz_on_sphere[1], xyz_on_sphere[0])
        elevation = np.arctan2(
            xyz_on_sphere[2], np.sqrt(xyz_on_sphere[0] ** 2 + xyz_on_sphere[1] ** 2)
        )

        def RzRy(azimuth, elevation):
            ca = np.cos(azimuth)
            sa = np.sin(azimuth)
            cb = np.cos(elevation)
            sb = np.sin(elevation)
            return np.array(
                [[ca * cb, -sa, ca * sb], [sa * cb, ca, sa * sb], [-sb, 0, cb]],
                dtype=np.float32,
            )

        point_on_sphere = (
            np.dot(RzRy(azimuth, HALF_PI - elevation), self.dot_verts) + xyz_on_sphere
        )
        self.cursor_dot_on_sphere.set_data_3d(*point_on_sphere)

        # Position of the line
        """ Line in 3D
        x / dxyz[0] = y / dxyz[1] = z / dxyz[2]
        ( z = 1.1 or -1.1  ) and ( dxyz[2] = +1 or -1 )
        """
        z = 1.1 if is_top else -1.1
        x = dxyz[0][0] * 1.1
        y = dxyz[1][0] * 1.1
        self.cursor_line.set_data_3d((0, x), (0, y), (0, z))
        # Color of the dot and line
        _c = "blue" if is_stable else "red"
        self.cursor_dot.set(color=_c)
        self.cursor_dot_on_sphere.set(color=_c)
        self.cursor_line.set(color=_c)


# class SamplesPatch3D:
#     def __init__(self, subplot, dot_radius=0.015):
#         self.ax = subplot
#         self.dot_radius = dot_radius
#         self.all_ICRs = []

#     def _