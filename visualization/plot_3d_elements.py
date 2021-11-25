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
    # arrow3d(ax, theta_x=-90, theta_z=0, offset=(0, -1.1, 0), color="black", alpha=0.5)
    # arrow3d(ax, theta_x=-90, theta_z=-90, offset=(-1.1, 0, 0), color="black", alpha=0.5)
    # arrow3d(ax, theta_x=0, theta_z=0, offset=(0, 0, -1.1), color="black", alpha=0.5)

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

    # Label
    # ax.xaxis.set_rotate_label(False)
    # ax.yaxis.set_rotate_label(False)
    # ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(r"$v_x$ [m/s]")
    ax.set_ylabel(r"$v_y$ [m/s]")
    ax.set_zlabel(r"$\omega$ [rad]", rotation=0)


"""
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
"""


class Line3D:
    def __init__(self, subplot, linewidth=1.0, color="black", alpha=1.0):
        # xs, ys, zs
        self.line = art3d.Line3D(
            (0, 0), (0, 0), (0, 0), linewidth=linewidth, color=color, alpha=alpha
        )
        subplot.add_line(self.line)


class MouseLocationPatch:
    def __init__(self, subplot):
        dot_size = 0.02
        samples = np.linspace(0, 2 * np.pi, 10)
        self.dot_verts = np.array(
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
        local_x, local_y = (event_xy[1], event_xy[0])
        x_from_centroid = local_x - local_centroid[0]
        y_from_centroid = local_y - local_centroid[1]
        is_top = y_from_centroid > 0
        dxyz = (
            ([y_from_centroid], [-x_from_centroid], [1])
            if is_top
            else ([-y_from_centroid], [x_from_centroid], [-1])
        )
        point_verts = self.dot_verts + dxyz
        self.cursor_dot.set_data_3d(*point_verts)
        # Position of the dot on the sphere
        """  x^2 + y^2 + z^2 = 1
        x = dxyz[0] * k
        y = dxyz[1] * k
        z = dxyz[2] * k
        ----
        k^2 * (dxyz[0]^2 + dxyz[1]^2 + dxyz[2]^2) = 1
        """
        k = 1.0 / np.sqrt(dxyz[0] ** 2 + dxyz[1] ** 2 + dxyz[2] ** 2)
        xyz_on_sphere = dxyz * k
        """
        (xyz_on_sphere)
        | z
        ----sqrt(x^2 + y^2)-----(origin)
        """
        azimuth = np.arctan2(xyz_on_sphere[1], xyz_on_sphere[0])
        elevation = np.arctan2(
            xyz_on_sphere[2], np.sqrt(xyz_on_sphere[0] ** 2 + xyz_on_sphere[1] ** 2)
        )
        # Rmat =
        # TODO

        point_verts = self.dot_verts * k + dxyz
        self.cursor_dot_on_sphere.set_data_3d(*point_verts)
        # Position of the line
        self.cursor_line.set_data_3d(
            (0, local_x), (0, local_y), (0, 0), linewidth=1.0, color="black"
        )

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
