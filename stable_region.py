import numpy as np
from tools import polygon_centroid, Tmat2D, TmatDot


def unit(rad):
    return np.array([np.cos(rad), np.sin(rad)])


class StableRegion:
    HALF_PI = np.pi / 2.0

    def __init__(self, default_mu=0.5):
        # Input coordinates
        self._xy_points = None
        self._line_contact_cases = None
        self._current_contact = None
        self._centroid = None  # Center of friction
        # foward_vector = (1, 0)

        # Transformation
        self._local_input_T = None
        self._input_local_T = None

        self._mu = default_mu

        # Pusher coordinates (centor of pusher)
        self._local_xy_points = None
        self._local_centroid = None
        self._local_lsupport = None  # xy
        self._local_rsupport = None  # xy

        # Stable conditions
        self._cond_FL1 = None
        self._cond_FL2 = None
        self._cond_FR1 = None
        self._cond_FR2 = None
        self._cond_WL1 = None
        self._cond_WL2 = None
        self._cond_WR1 = None
        self._cond_WR2 = None

    @property
    def line_contact_cases(self):
        return self._line_contact_cases

    @property
    def current_contact_idx(self):
        return self._current_contact

    @property
    def mu(self):
        return self._mu

    @property
    def local_input_Tmat(self):
        return self._local_input_T

    @property
    def local_xy_points(self):
        return self._local_xy_points

    @property
    def local_centroid(self):
        return self._local_centroid

    @property
    def local_lsupport(self):
        return self._local_lsupport

    @property
    def local_rsupport(self):
        return self._local_rsupport

    @property
    def stable_constraints_of_friction(self):
        return (self._cond_FL1, self._cond_FL2, self._cond_FR1, self._cond_FR2)

    @property
    def stable_constraints_of_wrench(self):
        return (self._cond_WL1, self._cond_WL2, self._cond_WR1, self._cond_WR2)

    def init_slider(self, xy_points_in_clockwise):
        """
        xy_points_in_clockwise: list of xy points in clockwise order. [(x,y), (x,y), ...]
        available_contact_cases: list of available contact cases. [0, 1, 2, ...]
        """
        if len(xy_points_in_clockwise) < 3:
            raise Exception("At least 3 points are needed to define a polygon.")
        if not isinstance(xy_points_in_clockwise, np.ndarray):
            xy_points_in_clockwise = np.array(xy_points_in_clockwise)
        self._xy_points = xy_points_in_clockwise
        self._line_contact_cases = len(self._xy_points)
        self.set_current_contact(0)

        # Friction center (cof) == center of gravity (centroid)
        self._centroid = polygon_centroid(self._xy_points)

        # Stable conditions
        self._init_stable_conditions()

    def set_current_contact(self, contact_index):
        def valid(idx):
            return idx % self._line_contact_cases

        idx = valid(contact_index)
        self._current_contact = idx

        # Support points in the input frame
        input_lsupport = self._xy_points[valid(idx + 1)]
        input_rsupport = self._xy_points[idx]
        input_localOrigin = (input_lsupport + input_rsupport) / 2.0
        input_localYvec = input_lsupport - input_rsupport
        input_localXvec = TmatDot(Tmat2D(-self.HALF_PI, 0, 0), input_localYvec)
        rad = np.arctan2(input_localXvec[1], input_localXvec[0])

        # Set pusher coordinates
        self._input_local_T = Tmat2D(rad, *input_localOrigin)
        self._local_input_T = np.linalg.inv(self._input_local_T)

        # Pusher coordinates
        self._local_xy_points = TmatDot(self._local_input_T, self._xy_points)
        self._local_centroid = polygon_centroid(self._local_xy_points)
        self._local_lsupport = TmatDot(self._local_input_T, input_lsupport)
        self._local_rsupport = TmatDot(self._local_input_T, input_rsupport)

        return self._current_contact

    def update_friction(self, mu):
        """
        mu: friction coefficient (mu = tan(alpha))
        """
        self._mu = mu
        self.update_friction_cone_stable()

    def _init_stable_conditions(self):
        """
        Reference: Page 159 of "Mechanics of robotic mnaipulation"
        (M. T. Mason, Mechanics of robotic manipulation. Cambridge, Mass: MIT Press, 2001.)
        """
        """
        # 1: Friction cone condition (mu dependent)
        -------------------------------------------------
        Left-hand ICR is stable if:
            y>= left cone의 -alpha (x=상수꼴이면 queryX<=x)
            y>= right cone의 alpha + 떨어진점 (x=상수꼴이면 queryX>=x)
        Right-hand ICR is stable if:
            y<= left cone의 -alpha + 떨어진점 (x=상수꼴이면 queryX>=x)
            y<= right cone의 alpha (x=상수꼴이면 queryX<=x)
        """
        self._cond_FL1 = LineConstraint(greater_than_y=True, greater_than_x=False)
        self._cond_FL2 = LineConstraint(greater_than_y=True, greater_than_x=True)
        self._cond_FR1 = LineConstraint(greater_than_y=False, greater_than_x=True)
        self._cond_FR2 = LineConstraint(greater_than_y=False, greater_than_x=False)
        self.update_friction_cone_stable()

        """
        # 2: Wrench condition (= Non-prehensile condition)) (shape dependent)
        -------------------------------------------------
        (r: centroid에서 가장 먼 cone까지의 거리, p: centroid에서 현재 cone까지의 거리)
        Left-hand ICR is stable if:
            y>= left cone과 centroid 사이의 수직이등분선 (x=상수꼴이면 queryX<=x)
            y>= `(현재)right cone->centroid`방향, centroid에서 r^2/p 거리 (x=상수꼴이면 queryX>=x)
        Right-hand ICR is stable if:
            y<= `(현재)left cone->centroid`방향, centroid에서 r^2/p 거리 (x=상수꼴이면 queryX>=x)
            y<= right cone과 centroid 사이의 수직이등분선 (x=상수꼴이면 queryX<=x)
        """
        # self.

    def update_friction_cone_stable(self):
        # Friction cone slope: m = tan(alpha) = friction coefficient mu
        alpha = np.arctan(self._mu)
        _rad1 = +alpha - self.HALF_PI
        _rad2 = -alpha + self.HALF_PI

        lsup = self._local_lsupport
        rsup = self._local_rsupport
        self._cond_FL1.update(lsup, _rad2)
        self._cond_FL2.update(self._farthest_local_xy(rsup, unit(alpha)), _rad1)
        self._cond_FR1.update(self._farthest_local_xy(lsup, unit(-alpha)), _rad2)
        self._cond_FR2.update(rsup, _rad1)

    def _farthest_local_xy(self, from_xy, direction):
        """
        from_xy: xy point in local frame
        direction: We need a farthest point in this direction.
        ---
        Find a point that maximizes the dot product with the direction vector..
        """
        farthest_point = None
        farthest_dot = 0.0
        for xy in self._local_xy_points:
            dot = np.dot(xy - from_xy, direction)
            if dot > farthest_dot:
                farthest_point = xy
                farthest_dot = dot
        return farthest_point


class LineConstraint:
    MAX_RAD = np.pi / 2.0

    def __init__(self, greater_than_y=True, greater_than_x=True):
        self._greater_than_y = greater_than_y
        self._greater_than_x = greater_than_x
        self._m = None
        self._b = None
        self._x1 = None
        self._is_caseA = None

    @property
    def greater_than_y(self):
        return self._greater_than_y

    @property
    def greater_than_x(self):
        return self._greater_than_x

    @property
    def is_caseA(self):
        return self._is_caseA

    @property
    def is_caseB(self):
        return not self._is_caseA

    @property
    def x1(self):
        return self._x1

    @property
    def standard_form(self):
        """ax + by + c = 0"""
        # case A: mx - y + b = 0
        # case B: x      - x1 = 0
        if self._is_caseA:
            return (self._m, -1.0, self._b)
        else:
            return (1.0, 0.0, -self._x1)

    def update(self, pass_point, angle_rad):
        """
        pass_point: (x1,y1)
        slope: m = tan(angle)
        Point-Slope form: y-y1 = m(x-x1)
        Slope-Intercept form: y = mx + b
        """
        self._m = np.tan(angle_rad)
        self._b = pass_point[1] - self._m * pass_point[0]
        self._x1 = pass_point[0]

        # -np.pi/2 <= angle_rad <= np.pi/2
        # case A: abs(angle_rad) < np.pi/2  => Check Y
        # case B: abs(angle_rad) == np.pi/2 => Check X
        abs_rad = abs(angle_rad)
        self._is_caseA = abs_rad < self.MAX_RAD
        if abs_rad > self.MAX_RAD:
            Exception("Angle must be in range of [-pi/2, pi/2]")

    def is_stable(self, xy):
        if self._is_caseA:
            return self._is_stable_caseA(xy)
        else:
            return self._is_stable_caseB(xy)

    def _is_stable_caseA(self, xy):
        if self._greater_than_y:
            return xy[1] >= self._m * xy[0] + self._b
        else:
            return xy[1] <= self._m * xy[0] + self._b

    def _is_stable_caseB(self, xy):
        if self._greater_than_x:
            return xy[0] >= self._x1
        else:
            return xy[0] <= self._x1
