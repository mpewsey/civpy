# spatial_hash_ex2.py
import numpy as np
from civpy.survey import SpatialHash
from matplotlib.patches import Rectangle, Circle

r = 30 # Query radius
np.random.seed(32874393)
x = np.random.uniform(-60, 60, (30000, 2))
s = SpatialHash(x, 10)
ax = s.plot(symbols=dict(points='r,'))

# Find points within radius of points
points = np.array([
    (30, -45),
    (60,  20)
])

# Plot query points
ax.plot(points[:,0], points[:,1], 'ko', zorder=10)

for p in points:
    # Plot found points
    i = s.query_point(p, r)
    xp = s.points[i]
    ax.plot(xp[:,0], xp[:,1], 'g,', zorder=5)

    # Plot circle
    circ = Circle(p, r, color='k', fill=False, zorder=10)
    ax.add_artist(circ)


# Find points within offset of range
ranges = np.array([
    [(-50, -50), (0, 60)]
])

for a, b in ranges:
    # Plot found points
    i = s.query_range(a, b, r)
    xp = s.points[i]
    ax.plot(xp[:,0], xp[:,1], 'g,')

    # Plot rectangles
    dx, dy = b - a
    c = np.linalg.norm(b - a)
    ang = np.arctan2(dy, dx) * 180/np.pi

    rect = Rectangle(a, c, r, angle=ang, color='k', fill=False, zorder=10)
    ax.add_artist(rect)

    rect = Rectangle(b, c, r, angle=ang+180, color='k', fill=False, zorder=10)
    ax.add_artist(rect)
