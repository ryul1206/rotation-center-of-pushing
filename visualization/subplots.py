from matplotlib.widgets import AxesWidget
from . import plot_2d_elements as p2d
from . import plot_3d_elements as p3d

"""
Reference:
https://github.com/matplotlib/matplotlib/blob/v3.5.0/lib/matplotlib/widgets.py#L1579-L1670
"""


class LeftPlot:
    """2D elements"""

    def __init__(self, ax, xlim, ylim, interval, useblit=True):
        """
        :param ax: matplotlib.axes.Axes (2D)
        :param xlim: tuple (xmin, xmax)
        :param ylim: tuple (ymin, ymax)
        :param interval: float (tick interval of xy-axis)
        """
        # blit
        self._awg = AxesWidget(ax)
        self._useblit = useblit and self._awg.canvas.supports_blit
        self._background = None
        # components
        p2d.build_static_elements(ax, xlim, ylim, interval)
        self.elem_slider = p2d.SliderPatch(ax)
        self.elem_constraints = p2d.ConstraintsPatch(ax, xlim, ylim)
        self.elem_cursor = p2d.MouseLocationPatch(ax)

    @property
    def widgetlock_available(self):
        return self._awg.canvas.widgetlock.available(self._awg)

    def on_clear(self, event):
        """Internal event handler to clear the left."""
        if self._awg.ignore(event):
            return
        self.elem_cursor.visibility(False)
        if self._useblit:
            self._background = self._awg.canvas.copy_from_bbox(self._awg.ax.bbox)

    def blit_draw(self):
        if self._useblit:
            if self._background is not None:
                self._awg.canvas.restore_region(self._background)
            self._awg.ax.draw_artist(self.elem_cursor.lineh)
            self._awg.ax.draw_artist(self.elem_cursor.linev)
            self._awg.ax.draw_artist(self.elem_cursor.cursor)
            self._awg.canvas.blit(self._awg.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False

    def update_constraints(self, sr):
        """
        :param sr: An instance of the stable_region.StableRegion class
        """
        self.elem_slider.update(sr)
        self.elem_constraints.update(sr)

    def update_cursor(
        self, event_xy, is_cursor_stable, is_cross_stable, is_cursor_locked
    ):
        """
        :param event_xy: tuple (x, y) in the figure frame
            == (y, x) in the pusher frame
        :param is_stable: bool
        """
        self.elem_cursor.update(
            event_xy, is_cursor_stable, is_cross_stable, is_cursor_locked
        )

    def remove_crosshair(self):
        self.elem_cursor.remove_crosshair()

    def cursor_visiblity(self, is_visible):
        self.elem_cursor.visibility(is_visible)


class RightPlot:
    """3D elements"""

    def __init__(self, ax, azim, elev, useblit=True):
        """
        :param ax: matplotlib.axes.Axes (3D)
        :param initial_azim: float (initial azimuth angle in degree)
        :param initial_elev: float (initial elevation angle in degree)
        """
        # blit
        self._awg = AxesWidget(ax)
        self._useblit = useblit and self._awg.canvas.supports_blit
        self._background = None
        # components
        self.initial_azim = azim
        self.initial_elev = elev
        self.reset_view()
        p3d.build_static_elements(ax)
        self.elem_constraints = p3d.ConstraintsPatch3D(ax)
        self.elem_cursor = p3d.MouseLocationPatch(ax)

    @property
    def widgetlock_available(self):
        return self._awg.canvas.widgetlock.available(self._awg)

    def on_clear(self, event):
        """Internal event handler to clear the left."""
        if self._awg.ignore(event):
            return
        self.elem_cursor.visibility(False)
        if self._useblit:
            self._background = self._awg.canvas.copy_from_bbox(self._awg.ax.bbox)

    def blit_draw(self):
        if self._useblit:
            if self._background is not None:
                self._awg.canvas.restore_region(self._background)
            self._awg.ax.draw_artist(self.elem_cursor.cursor_dot)
            self._awg.ax.draw_artist(self.elem_cursor.cursor_dot_on_sphere)
            self._awg.ax.draw_artist(self.elem_cursor.cursor_line)
            self._awg.canvas.blit(self._awg.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False

    def reset_view(self):
        self._awg.ax.azim = self.initial_azim
        self._awg.ax.elev = self.initial_elev

    def update_constraints(self, sr):
        """
        :param sr: An instance of the stable_region.StableRegion class
        """
        self.elem_constraints.update(sr)

    def update_cursor(self, event_xy, is_stable, is_cursor_locked):
        """
        :param event_xy: tuple (x, y) in the figure frame
            == (y, x) in the pusher frame
        :param is_stable: bool
        """
        self.elem_cursor.visibility(True)
        self.elem_cursor.update(event_xy, is_stable)

    def cursor_visiblity(self, is_visible):
        self.elem_cursor.visibility(is_visible)
