import matplotlib
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import numpy as np
import shapely.geometry as geom


"""
CAUTION:
(X, Y) -> (Y, X)
"""


def TmatDotBulk(Tmat, xys):
    _xys = np.vstack((xys.transpose(), np.ones(xys.shape[0])))
    res = np.dot(Tmat, _xys)[:-1].transpose()
    return res


def find_2d_line_std_form(p1, p2):
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0] * p1[1] - p1[0] * p2[1]
    return a, b, c


def find_intersection_of_2d_lines(line1_std_form, line2_std_form):
    a1, b1, c1 = line1_std_form
    a2, b2, c2 = line2_std_form
    determinant = a1 * b2 - a2 * b1
    if determinant != 0:
        x = (b1 * c2 - b2 * c1) / determinant
        y = (a2 * c1 - a1 * c2) / determinant
        return (x, y)
    else:  # Lines are parallel
        return None


def build_static_elements(ax, xlim=(-1, 1), ylim=(-1, 1), tick_interval=0.25):
    """2D"""
    ax.set_aspect("equal")
    # Show ti`cks in the left and lower axes only
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    offset = 0.0
    ax.spines["left"].set_position(("axes", 0.5 - offset))
    # ax.spines["bottom"].set_position("center")
    ax.spines["bottom"].set_position(("axes", 0.5 + offset))
    # Eliminate upper and right axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Label
    ax.set_ylabel("x", loc="top", rotation=0, labelpad=-12)
    ax.set_xlabel("y", loc="right")
    ax.invert_xaxis()

    # Tick
    xmin, xmax = xlim
    ymin, ymax = ylim
    ax.xaxis.set_ticks(np.arange(xmin, xmax, tick_interval))
    ax.yaxis.set_ticks(np.arange(ymin, ymax, tick_interval))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))


class MouseLocationPatch:
    def __init__(self, subplot):
        self.cursor = patches.Circle(
            (0, 0), radius=0.03, fill=False, edgecolor="gray", linewidth=2
        )
        subplot.add_patch(self.cursor)

    def update(self, event_xy, sr):
        """
        event_xy: (y, x) in local coordinates
        sr: An instance of the stable_region.StableRegion class
        """
        self.cursor.center = event_xy
        is_stable = sr.is_stable_in_local_frame((event_xy[1], event_xy[0]))
        self.cursor.set(color="blue" if is_stable else "red")
        return is_stable


class SliderPatch:
    def __init__(self, subplot):
        self.exterior = patches.Polygon(
            ((0, 0), (0, 0)), closed=True, fill=False, edgecolor="black", linewidth=1.5
        )
        self.centroid = patches.FancyArrowPatch(
            (0, 0), (0, 0), fill=True, color="black", linewidth=2
        )
        self.centroid.set_arrowstyle("-|>", head_length=4, head_width=2)
        subplot.add_patch(self.exterior)
        subplot.add_patch(self.centroid)
        self.axes = AxesLine(subplot)
        self.lfriction_cone = FrictionCone(subplot)
        self.rfriction_cone = FrictionCone(subplot)

    def update(self, sr):
        """
        sr: An instance of the stable_region.StableRegion class
        """
        # Slider outline
        self.exterior.set_xy(sr.local_xy_points[:, [1, 0]])  # CAUTION: xy -> yx
        # Slider centroid
        arrow_length = 0.23
        fw_xy = sr.local_forward_vector * arrow_length
        start_xy = sr.local_centroid
        end_xy = start_xy + fw_xy
        # CAUTION: xy -> yx
        self.centroid.set_positions((start_xy[1], start_xy[0]), (end_xy[1], end_xy[0]))
        # Slider axes
        self.axes.transformation(sr.local_input_Tmat)
        # Friction cones
        self.lfriction_cone.update(sr.local_lsupport, sr.mu)
        self.rfriction_cone.update(sr.local_rsupport, sr.mu)


class AxesLine:
    def __init__(self, subplot, linewidth=3, linelength=0.1):
        self._xoy = np.array([[linelength, 0], [0, 0], [0, linelength]])  # x, origin, y
        x, o, y = self._xoy
        self.xline = matplotlib.lines.Line2D(
            (o[1], x[1]), (o[0], x[0]), color="red", linewidth=linewidth
        )
        self.yline = matplotlib.lines.Line2D(
            (o[1], y[1]), (o[0], y[0]), color="green", linewidth=linewidth
        )
        subplot.add_line(self.xline)
        subplot.add_line(self.yline)

    def transformation(self, Tmat):
        x, o, y = TmatDotBulk(Tmat, self._xoy)
        self.xline.set_data((o[1], x[1]), (o[0], x[0]))
        self.yline.set_data((o[1], y[1]), (o[0], y[0]))


class FrictionCone:
    def __init__(self, subplot, linewidth=1, linelength=0.25):
        self.arrow_length = linelength
        self.arrow1 = patches.FancyArrowPatch(
            (0, 0), (0, 0), fill=False, color="red", linewidth=linewidth
        )
        self.arrow2 = patches.FancyArrowPatch(
            (0, 0), (0, 0), fill=False, color="red", linewidth=linewidth
        )
        self.arrow1.set_arrowstyle("->", head_length=4, head_width=2)
        self.arrow2.set_arrowstyle("->", head_length=4, head_width=2)
        subplot.add_patch(self.arrow1)
        subplot.add_patch(self.arrow2)

    def update(self, start_xy, mu):
        alpha = np.arctan(mu)
        dx = self.arrow_length * np.cos(alpha)
        dy = self.arrow_length * np.sin(alpha)
        end1_xy = start_xy + np.array([dx, dy])
        end2_xy = start_xy + np.array([dx, -dy])
        self.arrow1.set_positions((start_xy[1], start_xy[0]), (end1_xy[1], end1_xy[0]))
        self.arrow2.set_positions((start_xy[1], start_xy[0]), (end2_xy[1], end2_xy[0]))


class ConstraintsPatch:
    def __init__(self, subplot, xlim, ylim):
        self.elem_FL = LineConstraintPair(
            subplot, xlim, ylim, color="green", linewidth=1, alpha=0.15
        )
        self.elem_FR = LineConstraintPair(
            subplot, xlim, ylim, color="green", linewidth=1, alpha=0.15
        )
        self.elem_WL = LineConstraintPair(
            subplot, xlim, ylim, color="red", linewidth=1, alpha=0.1
        )
        self.elem_WR = LineConstraintPair(
            subplot, xlim, ylim, color="red", linewidth=1, alpha=0.1
        )

    def update(self, sr):
        """
        sr: An instance of the stable_region.StableRegion class
        """
        # (1) Friction cone condition
        fl1, fl2, fr1, fr2 = sr.stable_constraints_of_friction
        self.elem_FL.update(fl1, fl2)
        self.elem_FR.update(fr1, fr2)
        # (2) Non-prehensile condition
        wl1, wl2, wr1, wr2 = sr.stable_constraints_of_wrench
        self.elem_WL.update(wl1, wl2)
        self.elem_WR.update(wr1, wr2)


class LineConstraintPair:
    def __init__(
        self,
        subplot,
        matplot_xlim,
        matplot_ylim,
        color="black",
        linewidth=1,
        alpha=0.15,
    ):
        self.xlim = matplot_ylim
        self.ylim = matplot_xlim

        xs = (0, 0)
        ys = (0, 0)
        self.L1 = matplotlib.lines.Line2D(
            xs, ys, color=color, linewidth=linewidth, alpha=alpha * 2.0
        )
        self.L2 = matplotlib.lines.Line2D(
            xs, ys, color=color, linewidth=linewidth, alpha=alpha * 2.0
        )
        self.intersection = patches.Polygon(
            [xs], True, color=color, linewidth=None, alpha=alpha
        )
        subplot.add_line(self.L1)
        subplot.add_line(self.L2)
        subplot.add_patch(self.intersection)

    def update(self, line1, line2):
        """
        Intersection Polygon, NOT Triangle!
        (p1: points1, p2: points2, R: rect_vertices)

        P1 Polygon       P2 Polygon       Intersection Region
        R-p1--p2-----R   R-p1--p2-----R   R-p1--p2-----R   Y
        |##\ /       |   |  \ /#######|   |  \ /       |   |->X
        |###*        |   |   *########|   |   *        |
        |##/#\       |   |  /#\#######|   |  /#\       |
        |#/###\      | + | /###\######| = | /###\      | => Polygon (Many vertices)
        |/#####\     |   |/#####\#####|   |/#####\     |    when `is_up1` and
        p2######\    |   P2######\####|   P2######\    |    `is_up2` are False.
        |########\   |   |########\###|   |########\   |
        R---------p2-R   R---------p1-R   R---------p1-R

        (caseA) `slope!=inf` General Line => Check Y
        (caseB) `slope==inf` Vertical Line => Check X
        """
        p1_polygon = geom.Polygon(self._find_polygon(line1, self.L1))
        p2_polygon = geom.Polygon(self._find_polygon(line2, self.L2))
        intersection_polygon = p1_polygon.intersection(p2_polygon).exterior.coords[:]
        self._draw_intersection(intersection_polygon)

    def _find_polygon(self, line, matplot_L):
        # Find inside points
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        vertices = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        """ Index of edges and vertices
        1:(xmin, ymax)---1---2:(xmax, ymax)
            |                   |
            0                   2
            |                   |
        0:(xmin, ymin)---3---3:(xmax, ymin)
        """
        polygon_vertices = [False] * 4
        intersect_edges = [False] * 4  # The number of `True` must be 2.
        for i in range(4):
            if line.is_stable(vertices[i]):
                polygon_vertices[i] = True
                intersect_edges[i] = not intersect_edges[i]
                intersect_edges[i - 1] = not intersect_edges[i - 1]
        if intersect_edges.count(True) != 2:
            # All (all vertices are inside) or Nothing (no vertices are inside)
            return vertices if True in polygon_vertices else list()
        # Build polygon
        polygon = []
        intersection = []
        for i in range(4):
            if polygon_vertices[i]:
                polygon.append(vertices[i])
            if intersect_edges[i]:
                p1 = vertices[i]
                p2 = vertices[(i + 1) % 4]
                edge_std_form = find_2d_line_std_form(p1, p2)
                xy = find_intersection_of_2d_lines(line.standard_form, edge_std_form)
                polygon.append(xy)
                intersection.append(xy)
        # draw line
        self._draw_line(intersection[0], intersection[1], matplot_L)
        return polygon

    def _draw_line(self, point_a, point_b, matplot_L):
        plot_xs = (point_a[1], point_b[1])  # data ys
        plot_ys = (point_a[0], point_b[0])  # data xs
        matplot_L.set_data(plot_xs, plot_ys)

    def _draw_intersection(self, polygon_xys):
        if polygon_xys:
            yx = np.array(polygon_xys)[:, [1, 0]]
            self.intersection.set_xy(yx)
        else:
            self.intersection.set_xy([(0, 0)])


class SamplesPatch:
    def __init__(self, subplot, linewidth=1, linelength=0.25):
        self.ax = subplot
        self.arrow_width = linewidth
        self.arrow_length = linelength
        self.all_arrows = []

    def _add_arrow(self, n):
        """
        n: number of new arrows
        """
        new_arrows = [
            patches.FancyArrowPatch(
                (0, 0), (0, 0), fill=False, color="black", linewidth=self.arrow_width
            )
            for _ in range(n)
        ]
        for arrow in new_arrows:
            arrow.set_visible(False)
            arrow.set_arrowstyle("-|>", head_length=4, head_width=2)
            self.ax.add_patch(arrow)

    def update(self, stable_mask, samples=None):
        """
        stable_mask: [True, False, ...]
        samples: [[x, y, heading], ...]
        """
        req_n = len(stable_mask)
        cur_n = len(self.all_arrows)
        if cur_n < req_n:
            self._add_arrow(req_n - cur_n)
        elif cur_n > req_n:
            for arrow in self.all_arrows[req_n:]:
                arrow.set(visible=False)
        # Update
        for i in range(req_n):
            self.all_arrows[i].set(
                visible=True, color="blue" if stable_mask[i] else "red"
            )
            if samples is not None:
                x, y, heading = samples[i]
                dx = self.arrow_length * np.cos(heading)
                dy = self.arrow_length * np.sin(heading)
                self.all_arrows[i].set_positions((x, y), (x + dx, y + dy))
