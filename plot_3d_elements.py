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
        repeat(latitude, np.linspace(-np.pi / 2.0, np.pi / 2.0, odd))

    sphere(ax, "gray")

    # # x, y, z
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
