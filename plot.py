#!/usr/bin/python3
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button
import numpy as np

import stable_region
import plot_2d_elements as p2d
import plot_3d_elements as p3d

"""
CAUTION:
(X, Y) -> (Y, X)
"""


class PlotManager:
    def __init__(self, stable_region):
        self.sr = stable_region

        fig = plt.figure(figsize=(14, 8))
        fig.suptitle("Stable Region of Rotation Center for Pushing")
        gs = GridSpec(1, 2, width_ratios=[7, 5])

        # ======= LEFT FIGURES =======
        interval = 1.0
        limit = (3, 2)
        xlim = (-interval * limit[0], interval * limit[0] * 1.1)
        ylim = (-interval * limit[1], interval * limit[1] * 1.1)
        self.axL = fig.add_subplot(gs[0])
        p2d.build_static_elements(self.axL, xlim, ylim, interval)
        self.elem_slider = p2d.SliderPatch(self.axL)
        self.elem_constraints = p2d.ConstraintsPatch(self.axL, xlim, ylim)
        self.elem_mouse = p2d.MouseLocationPatch(self.axL)
        self._draw_stable_region()  # Initial drawing

        # ======= RIGTH FIGURES =======
        axR = fig.add_subplot(gs[1], projection="3d")
        p3d.build_static_elements(axR)

        # ======= WIDGETS =======
        # ax = plt.axes([left, bottom, width, height])
        self.friction_slider = Slider(
            ax=plt.axes([0.2, 0.08, 0.3, 0.03]),
            label=r"Input $\alpha$ [deg] (Friction $\mu=\tan(\alpha)$)",
            valmin=0.0,
            valmax=90.0,
            valinit=np.degrees(np.arctan(self.sr.mu)),
        )
        self.friction_slider.on_changed(self._update_friction)

        box = plt.axes([0.4, 0.02, 0.07, 0.04])
        reset_button = Button(box, "Reset All", hovercolor="0.95")
        reset_button.on_clicked(self._reset)

        box = plt.axes([0.22, 0.02, 0.05, 0.04])
        ccw_button = Button(box, "<< CCW", hovercolor="0.90")
        ccw_button.on_clicked(self._rotate_slider_ccw)

        box = plt.axes([0.28, 0.02, 0.05, 0.04])
        cw_button = Button(box, "CW >>", hovercolor="0.90")
        cw_button.on_clicked(self._rotate_slider_cw)

        plt.tight_layout(rect=[0, 0.03, 1, 1.05])
        plt.connect("motion_notify_event", self._mouse_update)
        plt.show()

    def _mouse_update(self, event):
        """
        When your mouse is on the left figure, event.inaxes is
            <class 'matplotlib.axes._subplots.AxesSubplot'>.
        When your mouse is on the right figure, event.inaxes is
            <class 'matplotlib.axes._subplots.Axes3DSubplot'>.
        """
        if isinstance(event.inaxes, self.axL.__class__):
            self.elem_mouse.update((event.xdata, event.ydata), self.sr)
            plt.draw()

    def _draw_stable_region(self):
        self.elem_slider.update(self.sr)
        self.elem_constraints.update(self.sr)

    def _update_friction(self, val):
        mu = np.tan(np.radians(self.friction_slider.val))
        self.sr.update_friction(mu)
        self._close_update()

    def _reset(self, event):
        self.friction_slider.reset()
        self.sr.set_current_contact(0)
        self.sr.update_shape()
        self._close_update()

    def _rotate_slider_ccw(self, event):
        idx = self.sr.current_contact_idx
        self.sr.set_current_contact(idx + 1)
        self.sr.update_shape()
        self._close_update()

    def _rotate_slider_cw(self, event):
        idx = self.sr.current_contact_idx
        self.sr.set_current_contact(idx - 1)
        self.sr.update_shape()
        self._close_update()

    def _close_update(self):
        """No logic HERE! Only drawing"""
        self._draw_stable_region()
        plt.draw()


if __name__ == "__main__":
    sr = stable_region.StableRegion(default_mu=0.5)

    input_points = [(-0.1, -0.1), (-0.1, 0.3), (0.4, 0.3), (0.4, 0.1), (0.2, -0.1)]
    # If points are in clockwise order, pusher will head opposite direction (from inside).
    # input_points.reverse()
    slider = np.array(input_points)
    sr.init_slider(slider)

    p = PlotManager(sr)
