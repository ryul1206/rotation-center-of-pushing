#!/usr/bin/python3
import matplotlib.pyplot as plt
from matplotlib import artist as martist
from matplotlib.backend_bases import MouseButton
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, AxesWidget
import numpy as np

from logic import stable_region
from visualization import subplots

# import sampling

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

        self.lastcursor_event_xy = (0, 0)
        self.crosshair_event_xy = (0, 0)
        self.is_cursor_locked = False

        """
        LEFT and RIGHT FIGURES
        """
        interval = 1.0
        limit = (3, 2)
        xlim = (-interval * limit[0], interval * limit[0] * 1.1)
        ylim = (-interval * limit[1], interval * limit[1] * 1.1)
        self.axL = fig.add_subplot(gs[0])
        self.axR = fig.add_subplot(gs[1], projection="3d")

        self.left_plot = subplots.LeftPlot(self.axL, xlim, ylim, interval)
        self.right_plot = subplots.RightPlot(self.axR, 30, 30)

        # Initial drawing
        self.left_plot.update_constraints(self.sr)
        self.right_plot.update_constraints(self.sr)
        plt.draw()

        """
        WIDGETS
        """
        self.widget_friction = Slider(
            ax=plt.axes([0.2, 0.09, 0.3, 0.04]),  # [left, bottom, width, height]
            label=r"Input $\alpha$ [deg] (Friction $\mu=\tan(\alpha)$)",
            valmin=0.0,
            valmax=90.0,
            valinit=np.degrees(np.arctan(self.sr.mu)),
        )
        self.widget_friction.on_changed(self._update_friction)

        box = plt.axes([0.4, 0.03, 0.07, 0.04])
        bt0 = Button(box, "Reset All", hovercolor="0.9")
        bt0.on_clicked(self._reset)

        box = plt.axes([0.26, 0.03, 0.05, 0.04])
        bt1 = Button(box, "<< CCW", hovercolor="0.9")
        bt1.on_clicked(self._rotate_slider_ccw)

        box = plt.axes([0.32, 0.03, 0.05, 0.04])
        bt2 = Button(box, "CW >>", hovercolor="0.9")
        bt2.on_clicked(self._rotate_slider_cw)

        box = plt.axes([0.048, 0.03, 0.035, 0.04])
        box.set_axis_off()
        box.text(0, 0.9, "*Click in the left figure to (un)lock the cursor.")
        self.cursor_lock_caution = box.text(
            0,
            0,
            "Cursor Locked",
            color="red",
            visible=False,
            fontsize=20,
            fontweight="bold",
        )
        # self.info = plt.axes([0.05, 0.03, 0.05, 0.04])
        # self.cursor_text = self.info.text(0.0, 0.0, "15", fontsize=12)

        # _ = Cursor(self.axL, useblit=True, color="red", linewidth=1)
        # _ = custom.CustomCursor(
        #     self.axL, self._on_mouse_move, useblit=True, color="red", linewidth=1
        # )
        # _ = custom.CustomCursor3D(self.axR, self._on_mouse_move, useblit=True)

        """
        EVENTS with blit
        """
        plt.connect("motion_notify_event", self._on_mouse_move)
        plt.connect("draw_event", self._on_clear)
        plt.connect("button_press_event", self._on_mouse_click)

        plt.tight_layout(rect=[0, 0.03, 1, 1.05])
        plt.show()

    def _on_clear(self, event):
        """Internal event handler to clear all"""
        self.left_plot.on_clear(event)
        self.right_plot.on_clear(event)

    def _update_cursor(self):
        is_cursor_stable = self.sr.is_stable_in_local_frame(
            (self.lastcursor_event_xy[1], self.lastcursor_event_xy[0])
        )
        is_cross_stable = self.sr.is_stable_in_local_frame(
            (self.crosshair_event_xy[1], self.crosshair_event_xy[0])
        )
        self.left_plot.update_cursor(
            self.crosshair_event_xy,
            is_cursor_stable,
            is_cross_stable,
            self.is_cursor_locked,
        )
        self.right_plot.update_cursor(
            self.lastcursor_event_xy, is_cursor_stable, self.is_cursor_locked
        )

    def _on_mouse_move(self, event):
        """
        When your mouse is on the left figure, event.inaxes is
            <class 'matplotlib.axes._subplots.AxesSubplot'>.
        When your mouse is on the right figure, event.inaxes is
            <class 'matplotlib.axes._subplots.Axes3DSubplot'>.
        """
        if (not self.left_plot.widgetlock_available) or (
            not self.right_plot.widgetlock_available
        ):
            return
        if event.inaxes == self.axL:
            cursor_xy = (event.xdata, event.ydata)
            local_xy = (cursor_xy[1], cursor_xy[0])
            is_stable = self.sr.is_stable_in_local_frame(local_xy)
            msg = "xy({:.3f}, {:.3f}):{}".format(
                local_xy[0], local_xy[1], "stable" if is_stable else "UNSTABLE"
            )
            print("Cursor {}".format(msg))
            self.crosshair_event_xy = cursor_xy
            if not self.is_cursor_locked:
                self.lastcursor_event_xy = cursor_xy
            self._update_cursor()
        else:
            self._update_cursor()
            self.left_plot.remove_crosshair()
        self._update_blit()

    def _update_blit(self):
        self.left_plot.blit_draw()
        self.right_plot.blit_draw()

    def _on_mouse_click(self, event):
        if isinstance(event.inaxes, self.axL.__class__):
            if event.button is MouseButton.LEFT:
                self._cursor_lock()
        self._close_update()

    def _cursor_lock(self, value=None):
        if value is None:
            self.is_cursor_locked = not self.is_cursor_locked
        else:
            self.is_cursor_locked = value
        self.cursor_lock_caution.set_visible(self.is_cursor_locked)

    def _update_friction(self, val):
        mu = np.tan(np.radians(self.widget_friction.val))
        self.sr.update_friction(mu)
        self._close_update()

    def _reset(self, event):
        self._cursor_lock(value=False)
        self.crosshair_event_xy = (0, 0)
        self.lastcursor_event_xy = (0, 0)
        self.sr.set_current_contact(0)
        self.widget_friction.reset()
        self.right_plot.reset_view()
        self._update_blit()
        self._close_update()

    def _rotate_slider_ccw(self, event):
        self.sr.set_current_contact(self.sr.current_contact_idx + 1)
        self._close_update()

    def _rotate_slider_cw(self, event):
        self.sr.set_current_contact(self.sr.current_contact_idx - 1)
        self._close_update()

    def _close_update(self):
        self.left_plot.cursor_visiblity(False)
        self.right_plot.cursor_visiblity(False)
        self.left_plot.update_constraints(self.sr)
        self.right_plot.update_constraints(self.sr)
        plt.draw()


if __name__ == "__main__":
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

    sr = stable_region.StableRegion(slider, default_mu=0.5)

    p = PlotManager(sr)
