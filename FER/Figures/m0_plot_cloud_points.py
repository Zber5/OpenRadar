import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import axes3d
import os

os.chdir("C:/Users/Zber/Documents/Dev_program/OpenRadar/FER/Figures")


class MyAxes3D(axes3d.Axes3D):

    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.w_zaxis.get_ticklines() + self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        # set visibility of some features False
        self.set_some_features_visibility(False)
        # draw the axes
        super(MyAxes3D, self).draw(renderer)
        # set visibility of some features True.
        # This could be adapted to set your features to desired visibility,
        # e.g. storing the previous values and restoring the values
        self.set_some_features_visibility(True)

        zaxis = self.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        # disable draw grid
        zaxis.axes._draw_grid = False

        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw:
            # draw zaxis on the left side
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw:
            # draw zaxis on the right side
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2],
                             tmp_planes[1], tmp_planes[0],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes

        # disable draw grid
        zaxis.axes._draw_grid = draw_grid_old


path = 'fig_data/clouds_points.npy'
data = np.load(path)

mask_t = (data[0] == 1)
mask_tt = (data[0] == 5)
mask_ttt = (data[0] == 10)

z_t = -data[1, mask_t]
y_t = data[2, mask_t]
x_t = data[3, mask_t]

z_tt = -data[1, mask_tt]
y_tt = data[2, mask_tt]
x_tt = data[3, mask_tt]

z_ttt = -data[1, mask_ttt]
y_ttt = data[2, mask_ttt]
x_ttt = data[3, mask_ttt]

pt = (-1 < z_t) & (z_t < 1) & (-2 < x_t) & (x_t < 2) & (0 < y_t) & (y_t < 5)
ptt = (-1 < z_tt) & (z_tt < 1) & (-2 < x_tt) & (x_tt < 2) & (0 < y_tt) & (y_tt < 5)
pttt = (-1 < z_ttt) & (z_ttt < 1) & (-2 < x_ttt) & (x_ttt < 2) & (0 < y_ttt) & (y_ttt < 5)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect([5, 4, 2])

# ax.scatter(xs=x_t[pt], ys=y_t[pt], zs=z_t[pt], c='tab:blue', label='t-1')
# ax.scatter(xs=x_tt[ptt], ys=y_tt[ptt], zs=z_tt[ptt], c='tab:orange', label='t')
# ax.scatter(xs=x_ttt[pttt], ys=y_ttt[pttt], zs=z_ttt[pttt], c='tab:green', label='t+1')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.scatter(xs=y_t[pt], ys=x_t[pt], zs=z_t[pt], s=60, color='orange', label='t-1', alpha=1)
ax.scatter(xs=y_tt[ptt], ys=x_tt[ptt], zs=z_tt[ptt], s=60, color='limegreen', label='t', alpha=1)
ax.scatter(xs=y_ttt[pttt], ys=x_ttt[pttt], zs=z_ttt[pttt], s=60, color='royalblue', label='t+1', alpha=1)

ax.set_ylim3d(-2, 2)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.xaxis.set_major_locator(ticker.MaxNLocator(3))


ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xlim3d(0, 5)
# plt.gca().invert_xaxis()

ax.zaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_zlim3d(-1, 1)

ax.set_ylabel("Y (m)", fontsize=50)
ax.set_xlabel("X (m)", fontsize=50)
ax.set_zlabel("Z (m)", fontsize=50)

ax.xaxis.labelpad = 50
ax.yaxis.labelpad = 50
ax.zaxis.labelpad = 30

ax.xaxis.set_tick_params(labelsize=35, length=5, direction='in', pad=20)
ax.yaxis.set_tick_params(labelsize=35, length=5, direction='in', pad=20)
ax.zaxis.set_tick_params(labelsize=35, length=5, direction='in', pad=20)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(1.5)

ax.legend()
ax.legend(loc="upper right", prop={'size': 30}, ncol=1, bbox_to_anchor=(0.25, 0.55), fancybox=True,
          labelspacing=0.1, handletextpad=0.3, columnspacing=0.5, handlelength=1.1)

plt.tight_layout()
plt.show()
