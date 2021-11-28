#!/usr/bin/python3
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button
import numpy as np

from logic import stable_region
from visualization import plot_2d_elements as p2d
from visualization import plot_3d_elements as p3d

"""
CAUTION:
(X, Y) -> (Y, X)
"""


class PlotManager:
    def __init__(self, stable_region):
        self.sr = stable_region

        fig = plt.figure(figsize=(16, 8))
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
        self.elem_mouse2d = p2d.MouseLocationPatch(self.axL)

        # ======= RIGTH FIGURES =======
        self.initial_azim_elev = (30, 30)  # deg
        self.axR = fig.add_subplot(gs[1], projection="3d")
        self.axR.azim = self.initial_azim_elev[0]
        self.axR.elev = self.initial_azim_elev[1]
        p3d.build_static_elements(self.axR)
        self.elem_constraints3d = p3d.ConstraintsPatch3D(self.axR)
        self.last_cursor_event_xy = (0, 0)
        self.is_cursor_locked = False
        self.elem_mouse3d = p3d.MouseLocationPatch(self.axR)

        # Initial drawing
        self._close_update()

        # ======= WIDGETS =======
        # ax = plt.axes([left, bottom, width, height])
        self.friction_slider = Slider(
            ax=plt.axes([0.2, 0.09, 0.3, 0.04]),
            label=r"Input $\alpha$ [deg] (Friction $\mu=\tan(\alpha)$)",
            valmin=0.0,
            valmax=90.0,
            valinit=np.degrees(np.arctan(self.sr.mu)),
        )
        self.friction_slider.on_changed(self._update_friction)

        box = plt.axes([0.4, 0.03, 0.07, 0.04])
        reset_button = Button(box, "Reset All", hovercolor="0.95")
        reset_button.on_clicked(self._reset)

        box = plt.axes([0.26, 0.03, 0.05, 0.04])
        ccw_button = Button(box, "<< CCW", hovercolor="0.90")
        ccw_button.on_clicked(self._rotate_slider_ccw)

        box = plt.axes([0.32, 0.03, 0.05, 0.04])
        cw_button = Button(box, "CW >>", hovercolor="0.90")
        cw_button.on_clicked(self._rotate_slider_cw)

        # ======= EVENTS =======
        box = plt.axes([0.05, 0.03, 0.035, 0.04])
        box.set_axis_off()
        box.text(0, 0.9, "Click in the left figure to (un)lock the cursor.")
        self.cursor_lock_caution = box.text(
            0,
            0,
            "Cursor Locked",
            color="red",
            visible=False,
            fontsize=20,
            fontweight="bold",
        )
        plt.connect("motion_notify_event", self._on_mouse_move)
        plt.connect("button_press_event", self._on_mouse_click)
        plt.tight_layout(rect=[0, 0.03, 1, 1.05])
        plt.show()

    def _on_mouse_move(self, event):
        """
        When your mouse is on the left figure, event.inaxes is
            <class 'matplotlib.axes._subplots.AxesSubplot'>.
        When your mouse is on the right figure, event.inaxes is
            <class 'matplotlib.axes._subplots.Axes3DSubplot'>.
        """
        if isinstance(event.inaxes, self.axL.__class__):
            if not self.is_cursor_locked:
                self.last_cursor_event_xy = (event.xdata, event.ydata)
                print(
                    "Cursor (x: {:.3f}, y: {:.3f})".format(*self.last_cursor_event_xy)
                )
                self._update_last_cursor()
                plt.draw()

    def _on_mouse_click(self, event):
        if isinstance(event.inaxes, self.axL.__class__):
            if event.button is MouseButton.LEFT:
                self.is_cursor_locked = not self.is_cursor_locked
                self.cursor_lock_caution.set_visible(self.is_cursor_locked)
                plt.draw()

    def _update_last_cursor(self):
        is_stable = self.elem_mouse2d.update(self.last_cursor_event_xy, self.sr)
        self.elem_mouse3d.update(
            self.last_cursor_event_xy, self.sr.local_centroid, is_stable
        )

    def _update_friction(self, val):
        mu = np.tan(np.radians(self.friction_slider.val))
        self.sr.update_friction(mu)
        self._close_update()

    def _reset(self, event):
        self.axR.azim = self.initial_azim_elev[0]
        self.axR.elev = self.initial_azim_elev[1]
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
        self.elem_slider.update(self.sr)
        self.elem_constraints.update(self.sr)
        self.elem_constraints3d.update(self.sr)
        self._update_last_cursor()
        plt.draw()


if __name__ == "__main__":
    sr = stable_region.StableRegion(default_mu=0.5)

    input_points = [
        (-0.1, -0.1),
        (-0.12, 0.1),
        (0.0, 0.3),
        (0.4, 0.3),
        (0.4, 0.1),
        (0.2, -0.1),
    ]
    # If points are in clockwise order, pusher will head opposite direction (from inside).
    # input_points.reverse()
    slider = np.array(input_points)
    sr.init_slider(slider)

    p = PlotManager(sr)
