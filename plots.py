import matplotlib.pyplot as plt
import numpy as np

method = 'nothing_next'

fig, ax = plt.subplots(figsize = (9, 9))

xs = []
ys = []
for x in np.linspace(0, 1, 5):
    for y in np.linspace(0,1, 5):
        xs.append(x)
        ys.append(y)
if method != 'nothing_next':
    ax.plot(xs, ys, 'ko', ms = 20, mfc = 'white', markeredgewidth = 3, zorder = 0)
else:
    ax.plot(xs, ys, 'ko', ms = 20, mfc = 'red', markeredgewidth = 3, zorder = 0)

if  'nothing' not in method:
    ax.plot(0.5, 0.5, 'ko',  ms = 30, mfc = 'red', markeredgewidth = 5, zorder = 1)

for i in range(len(xs)-1):
    ax.plot([xs[i], xs[i+1]], [0, 0], 'b-', lw = 2, zorder = -1)
    ax.plot([xs[i], xs[i+1]], [1, 1], 'b-', lw = 2, zorder = -1)
for i in range(len(ys)-1):
    ax.plot([0, 0], [ys[i], ys[i+1]], 'b-', lw = 2, zorder = -1)
    ax.plot([1, 1], [ys[i], ys[i+1]], 'b-', lw = 2, zorder = -1)

if method == 'explicit':
    ax.plot([0.5, 0.5], [0.25, 0.75], 'k-', lw=5, zorder=-1)
    ax.plot( [0.25, 0.75],[0.5, 0.5], 'k-', lw=5, zorder=-1)
elif method == 'implicit':
    xs = [0.25, 0.75, 0.5, 0.5]
    ys = [0.5, 0.5, 0.25, 0.75]
    ax.plot(xs, ys, 'ko', ms=20, mfc='red', markeredgewidth=3, zorder=0)
    ax.plot([0.5, 0.5], [0.25, 0.75], 'r-', lw=5, zorder=-1)
    ax.plot( [0.25, 0.75],[0.5, 0.5], 'r-', lw=5, zorder=-1)

elif method == 'ADI_1':
    xs = [0.25, 0.75]
    ys = [0.5, 0.5]
    ax.plot(xs, ys, 'ko', ms=20, mfc='red', markeredgewidth=3, zorder=0)
    ax.plot([0.5, 0.5], [0.25, 0.75], 'k-', lw=5, zorder=-1)
    ax.plot( [0.25, 0.75],[0.5, 0.5], 'r-', lw=5, zorder=-1)

elif method == 'ADI_2':
    xs = [0.5, 0.5]
    ys = [0.25, 0.75]
    ax.plot(xs, ys, 'ko', ms=20, mfc='red', markeredgewidth=3, zorder=0)
    ax.plot([0.5, 0.5], [0.25, 0.75], 'r-', lw=5, zorder=-1)
    ax.plot( [0.25, 0.75],[0.5, 0.5], 'k-', lw=5, zorder=-1)

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='k', lw=4),
                Line2D([0], [0], color='r', lw=4)]
# ax.legend(custom_lines,  ['boundary conditions', 'old timestep', 'next timestep'], loc ='upper center', fontsize = 'x-large', ncol=3)

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.axis('off')
plt.tight_layout()
plt.savefig(method + '.pdf',  bbox_inches = 0)


plt.show()