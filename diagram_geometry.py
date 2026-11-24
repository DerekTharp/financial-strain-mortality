"""Boxes and edge-anchored arrows for the supplementary diagrams, in points."""

from dataclasses import dataclass
from math import hypot

from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Polygon


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    def edge(self, side):
        left, bottom = self.x - self.width / 2, self.y - self.height / 2
        if side in ('top', 'bottom'):
            return self.x, bottom + (self.height if side == 'top' else 0)
        if side in ('left', 'right'):
            return left + (self.width if side == 'right' else 0), self.y
        raise ValueError(side)

    def toward(self, other):
        """The point where the straight line to the centre of another box crosses this box's edge.
        The rounding of the corners is ignored."""
        dx, dy = other.x - self.x, other.y - self.y
        scale = 1 / max(2 * abs(dx) / self.width, 2 * abs(dy) / self.height)
        return self.x + scale * dx, self.y + scale * dy

    def draw(self, ax, label, *, size, weight='normal', italic=False, ink='#191919', **border):
        left, _ = self.edge('left')
        _, bottom = self.edge('bottom')
        patch = FancyBboxPatch((left, bottom), self.width, self.height,
                               boxstyle='round,pad=0,rounding_size=3', **border)
        ax.add_patch(patch)
        ax.text(self.x, self.y, label, ha='center', va='center', fontsize=size,
                fontfamily='DejaVu Sans', fontweight=weight, linespacing=1.10,
                fontstyle='italic' if italic else 'normal', color=ink)


def canvas(width, height):
    fig = Figure(figsize=(width / 72, height / 72), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.set_axis_off()
    return fig, ax


def line(ax, points, color, width=1.1, style='-'):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=width, linestyle=style,
            solid_capstyle='butt', dash_capstyle='round', solid_joinstyle='round', zorder=1)


def arrow(ax, points, color, width=1.1, style='-'):
    """Put the tip on the endpoint and join the shaft just inside the head."""
    (x0, y0), (x, y) = points[-2:]
    length = hypot(x - x0, y - y0)
    ux, uy = (x - x0) / length, (y - y0) / length
    head, half = 4.7, 2.25
    bx, by = x - head * ux, y - head * uy
    line(ax, [*points[:-1], (bx + 0.65 * ux, by + 0.65 * uy)], color, width, style)
    ax.add_patch(Polygon([(x, y), (bx - half * uy, by + half * ux),
                         (bx + half * uy, by - half * ux)],
                         facecolor=color, edgecolor='none', zorder=2))


def link(ax, source, target, color, width=1.1, style='-'):
    """A straight arrow from the edge of one box to the edge of another, on the line through their centres."""
    arrow(ax, [source.toward(target), target.toward(source)], color, width, style)


def save(fig, folder, stem):
    fig.savefig(folder / f'{stem}.png', dpi=400)
    fig.savefig(folder / f'{stem}.pdf')
