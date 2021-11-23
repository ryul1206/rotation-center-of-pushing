#!/usr/bin/python3
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button
import numpy as np
from tools import TmatDot
import stable_region

# import time

"""
CAUTION:
(X, Y) -> (Y, X)
"""


class AxesLine:
    def __init__(self, linewidth=3, linelength=0.1):
        self._xoy = np.array([[linelength, 0], [0, 0], [0, linelength]])  # x, origin, y
        x, o, y = self._xoy
        self.xline = matplotlib.lines.Line2D(
            (o[1], x[1]), (o[0], x[0]), color="red", linewidth=linewidth
        )
        self.yline = matplotlib.lines.Line2D(
            (o[1], y[1]), (o[0], y[0]), color="green", linewidth=linewidth
        )

    def transformation(self, Tmat):
        x, o, y = TmatDot(Tmat, self._xoy)
        self.xline.set_data((o[1], x[1]), (o[0], x[0]))
        self.yline.set_data((o[1], y[1]), (o[0], y[0]))


class FrictionCone:
    def __init__(self, linewidth=1, linelength=0.25):
        self.arrow_length = linelength
        self.arrow1 = patches.FancyArrowPatch(
            (0, 0), (0, 0), fill=False, color="red", linewidth=linewidth
        )
        self.arrow2 = patches.FancyArrowPatch(
            (0, 0), (0, 0), fill=False, color="red", linewidth=linewidth
        )
        self.arrow1.set_arrowstyle("-|>", head_length=4, head_width=2)
        self.arrow2.set_arrowstyle("-|>", head_length=4, head_width=2)

    def update(self, start_xy, mu):
        alpha = np.arctan(mu)
        dx = self.arrow_length * np.cos(alpha)
        dy = self.arrow_length * np.sin(alpha)
        end1_xy = start_xy + np.array([dx, dy])
        end2_xy = start_xy + np.array([dx, -dy])
        self.arrow1.set_positions((start_xy[1], start_xy[0]), (end1_xy[1], end1_xy[0]))
        self.arrow2.set_positions((start_xy[1], start_xy[0]), (end2_xy[1], end2_xy[0]))


class LineConstraintPair:
    def __init__(self, matplot_xlim, matplot_ylim, color="black", linewidth=1):
        self.xlim = matplot_ylim
        self.ylim = matplot_xlim

        xs = (0, 0)
        ys = (0, 0)
        self.L1 = matplotlib.lines.Line2D(xs, ys, color=color, linewidth=linewidth)
        self.L2 = matplotlib.lines.Line2D(xs, ys, color=color, linewidth=linewidth)

    def update(self, line1, line2):
        """
        ax + by + c = 0
        line1: (a1, b1, c1)
        line2: (a2, b2, c2)
        """
        a1, b1, c1 = line1.standard_form
        a2, b2, c2 = line2.standard_form
        # Lines
        points1 = self._draw_line(a1, b1, c1, self.L1)
        points2 = self._draw_line(a2, b2, c2, self.L2)
        # Intersection Region (Polygon, NOT Triangle)
        is_up1 = line1.greater_than_y
        is_up2 = line2.greater_than_y
        is_rh1 = line1.greater_than_x
        is_rh2 = line2.greater_than_x



        determinant = a1 * b2 - a2 * b1
        if determinant != 0:
            # 
            Minv = np.array([[b2, -b1], [-a2, a1]]) / determinant
            intersection_xy = np.dot(Minv, np.array([c1, c2]))
        else:
            # lines are parallel
            pass

    def _draw_line(self, a, b, c, L):
        """
        y = -a/b * x - c/b
        vertical line: b = 0
        horizontal line: a = 0
        """
        minx, maxx = self.xlim
        miny, maxy = self.ylim
        if b != 0.0:
            # y = -a/b * x - c/b
            _m = -a / b
            _k = -c / b
            start = (minx, _m * minx + _k)
            end = (maxx, _m * maxx + _k)
            if a != 0.0:
                # x = -b/a * y - c/a
                _m = -b / a
                _k = -c / a
                if start[1] < miny:
                    start = (_m * miny + _k, miny)
                if end[1] > maxy:
                    end = (_m * maxy + _k, maxy)
        else:
            _x = -c / a
            start = (_x, miny)  # (_x, -np.inf)
            end = (_x, maxy)  # (_x, np.inf)
        L.set_data((start[1], end[1]), (start[0], end[0]))
        return (start, end)


class PlotManager:
    def __init__(self, stable_region):
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle("Stable Region of Rotation Center for Pushing")
        gs = GridSpec(1, 2, width_ratios=[7, 5])

        self.sr = stable_region
        # Define initial parameter
        init_friction = self.sr.mu

        # ======= LEFT FIGURES =======
        interval = 1.0
        limit = (3, 2)
        xlim = (-interval * limit[0], interval * limit[0] * 1.1)
        ylim = (-interval * limit[1], interval * limit[1] * 1.1)
        self.axL = fig.add_subplot(gs[0])
        # Elements
        self.elem_slider = patches.Polygon(
            ((0, 0), (0, 0)), closed=True, fill=False, edgecolor="black", linewidth=1.5
        )
        self.axL.add_patch(self.elem_slider)

        self.elem_slider_axes = AxesLine()
        self.axL.add_line(self.elem_slider_axes.xline)
        self.axL.add_line(self.elem_slider_axes.yline)

        self.elem_lfriction_cone = FrictionCone()
        self.elem_rfriction_cone = FrictionCone()
        self.axL.add_patch(self.elem_lfriction_cone.arrow1)
        self.axL.add_patch(self.elem_lfriction_cone.arrow2)
        self.axL.add_patch(self.elem_rfriction_cone.arrow1)
        self.axL.add_patch(self.elem_rfriction_cone.arrow2)

        self.elem_FL = LineConstraintPair(xlim, ylim, color="green", linewidth=1)
        self.elem_FR = LineConstraintPair(xlim, ylim, color="green", linewidth=1)
        self.axL.add_line(self.elem_FL.L1)
        self.axL.add_line(self.elem_FL.L2)
        self.axL.add_line(self.elem_FR.L1)
        self.axL.add_line(self.elem_FR.L2)

        self._static_figure(self.axL, xlim, ylim, interval)
        self._draw_stable_region()

        # ======= RIGTH FIGURES =======
        self.axR = fig.add_subplot(gs[1], projection="3d")
        self._velocity_sphere(self.axR)

        # ======= WIDGETS =======
        # Make a horizontal slider to control the friction.
        axfriction = plt.axes([0.2, 0.08, 0.3, 0.03])  # left, bottom, width, height
        self.friction_slider = Slider(
            ax=axfriction,
            label=r"Input $\alpha$ [deg] (Friction $\mu=\tan(\alpha)$)",
            valmin=0.0,
            valmax=90.0,
            valinit=np.degrees(np.arctan(init_friction)),
        )
        self.friction_slider.on_changed(self._update_friction)

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        reset_ax = plt.axes([0.4, 0.02, 0.07, 0.04])  # left, bottom, width, height
        reset_button = Button(reset_ax, "Reset All", hovercolor="0.95")
        reset_button.on_clicked(self._reset)

        # Create a `matplotlib.widgets.Button` to change the contact line.
        ccw_ax = plt.axes([0.22, 0.02, 0.05, 0.04])  # left, bottom, width, height
        ccw_button = Button(ccw_ax, "<< CCW", hovercolor="0.95")
        ccw_button.on_clicked(self._rotate_slider_ccw)
        cw_ax = plt.axes([0.28, 0.02, 0.05, 0.04])  # left, bottom, width, height
        cw_button = Button(cw_ax, "CW >>", hovercolor="0.95")
        cw_button.on_clicked(self._rotate_slider_cw)

        plt.tight_layout(rect=[0, 0.03, 1, 1.05])
        plt.show()

    @staticmethod
    def _static_figure(ax, xlim=(-1, 1), ylim=(-1, 1), tick_interval=0.25):
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

    @staticmethod
    def _velocity_sphere(ax):
        """3D"""
        # Sphere
        def longitude(alpha):
            """Longitude (vertical lines)"""
            dt = np.linspace(0, 2 * np.pi, 100)
            sphere_radius = 1.0
            _x = sphere_radius * np.cos(dt)
            z = sphere_radius * np.sin(dt)
            x = _x * np.cos(alpha)
            y = _x * np.sin(alpha)
            return (x, y, z)

        def latitude(beta):
            """Latitude (horizontal lines)"""
            dt = np.linspace(0, 2 * np.pi, 100)
            sphere_radius = 1.0
            radius = sphere_radius * np.cos(beta)
            x = radius * np.cos(dt)
            y = radius * np.sin(dt)
            z = (sphere_radius * np.sin(beta)) * np.ones(len(dt))
            return (x, y, z)

        def sphere(ax, color, transparency, width):
            initial = True
            odd = 13  # 15 degrees
            for alpha in np.linspace(0, np.pi, odd):
                if initial:
                    initial = False
                else:
                    x, y, z = longitude(alpha)
                    ax.plot(x, y, z, color=color, alpha=transparency, linewidth=width)
            initial = True
            for beta in np.linspace(-np.pi / 2.0, np.pi / 2.0, odd):
                if initial:
                    initial = False
                else:
                    x, y, z = latitude(beta)
                    ax.plot(x, y, z, color=color, alpha=transparency, linewidth=width)

        sphere(ax, "gray", 0.5, 1)

        # Custom axis
        def arrow3d(
            ax,
            length=2.3,
            width=0.005,
            head=0.05,
            headwidth=3,
            theta_x=0,
            theta_z=0,
            offset=(0, 0, 0),
            **kw
        ):
            w = width
            h = head
            hw = headwidth
            theta_x = np.deg2rad(theta_x)
            theta_z = np.deg2rad(theta_z)

            a = [
                [0, 0],
                [w, 0],
                [w, (1 - h) * length],
                [hw * w, (1 - h) * length],
                [0, length],
            ]
            a = np.array(a)

            r, theta = np.meshgrid(a[:, 0], np.linspace(0, 2 * np.pi, 30))
            z = np.tile(a[:, 1], r.shape[0]).reshape(r.shape)
            x = r * np.sin(theta)
            y = r * np.cos(theta)

            rot_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)],
                ]
            )
            rot_z = np.array(
                [
                    [np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1],
                ]
            )

            b1 = np.dot(rot_x, np.c_[x.flatten(), y.flatten(), z.flatten()].T)
            b2 = np.dot(rot_z, b1)
            b2 = b2.T + np.array(offset)
            x = b2[:, 0].reshape(r.shape)
            y = b2[:, 1].reshape(r.shape)
            z = b2[:, 2].reshape(r.shape)
            ax.plot_surface(x, y, z, **kw)

        # x, y, z
        arrow3d(
            ax, theta_x=-90, theta_z=0, offset=(0, -1.1, 0), color="black", alpha=0.5
        )
        arrow3d(
            ax, theta_x=-90, theta_z=-90, offset=(-1.1, 0, 0), color="black", alpha=0.5
        )
        arrow3d(ax, theta_x=0, theta_z=0, offset=(0, 0, -1.1), color="black", alpha=0.5)

        # Make axes limits
        xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
        XYZlim = [min(xyzlim[0]), max(xyzlim[1])]
        ax.set_xlim3d(XYZlim)
        ax.set_ylim3d(XYZlim)
        ax.set_zlim3d(XYZlim)
        ax.set_box_aspect((1, 1, 1))
        ax.grid(False)

        # Label
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.zaxis.set_rotate_label(False)
        ax.set_xlabel(r"$v_x$")
        ax.set_ylabel(r"$v_y$")
        ax.set_zlabel(r"$\omega$", rotation=0)

        # =========== DATA ===========

    def _draw_stable_region(self):
        # Slider outline
        xys = self.sr.local_xy_points[:, [1, 0]]  # CAUTION: xy
        self.elem_slider.set_xy(xys)
        # Slider axes
        self.elem_slider_axes.transformation(self.sr.local_input_Tmat)
        # Friction cones
        mu = self.sr.mu
        self.elem_lfriction_cone.update(self.sr.local_lsupport, mu)
        self.elem_rfriction_cone.update(self.sr.local_rsupport, mu)
        # (1) Friction cone condition
        fl1, fl2, fr1, fr2 = self.sr.stable_constraints_of_friction
        self.elem_FL.update(fl1, fl2)
        self.elem_FR.update(fr1, fr2)
        # (2) Non-prehensile condition

    def _update_friction(self, val):
        mu = np.tan(np.radians(self.friction_slider.val))
        self.sr.update_friction(mu)
        self._close_update()

    def _reset(self, event):
        self.friction_slider.reset()
        self.sr.set_current_contact(0)
        self._close_update()

    def _rotate_slider_ccw(self, event):
        idx = self.sr.current_contact_idx
        self.sr.set_current_contact(idx + 1)
        self._close_update()

    def _rotate_slider_cw(self, event):
        idx = self.sr.current_contact_idx
        self.sr.set_current_contact(idx - 1)
        self._close_update()

    def _close_update(self):
        self.sr.update_friction_cone_stable()
        self._draw_stable_region()
        plt.draw()


if __name__ == "__main__":
    sr = stable_region.StableRegion(default_mu=0.5)

    slider = np.array([(-0.1, -0.1), (-0.1, 0.3), (0.4, 0.3), (0.4, 0.1), (0.2, -0.1)])
    sr.init_slider(slider)

    p = PlotManager(sr)
