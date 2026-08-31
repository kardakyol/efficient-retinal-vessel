import math
import os

OUT = os.environ.get("FIG_OUT", "outputs/figures")
os.makedirs(OUT, exist_ok=True)

INK   = "#1f2733"
MUTE  = "#5b6b7d"
PANEL = "#f6f8fb"
EDGE  = "#c7d2e0"
BLUE  = "#2563a6"   # data
GREEN = "#2e8b57"   # full sup
ORANGE= "#d2691e"   # hrf
TEAL  = "#0e8388"   # density
PURPLE= "#6a4c93"   # granulometry
CRIM  = "#b83b5e"   # feedback
AMBER = "#d99000"   # growth/self-label
FONT  = "Helvetica, Arial, sans-serif"

def _esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.parts = []

    def rect(self, x, y, w, h, fill="none", stroke=INK, sw=1.5, rx=8, dash=None, opacity=1):

        d = f'stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>')
        
    def line(self, x1, y1, x2, y2, stroke=INK, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')
        
    def circle(self, cx, cy, r, fill="none", stroke=INK, sw=1.5, opacity=1):
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')
        
    def text(self, x, y, s, size=13, anchor="middle", fill=INK, weight="normal", italic=False, family=FONT):
        it = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}"{it}>{_esc(s)}</text>')
        
    def textlines(self, x, y, lines, size=12, anchor="middle", fill=INK, weight="normal", lh=15, italic=False):
        for i, ln in enumerate(lines):
            self.text(x, y + i*lh, ln, size, anchor, fill, weight, italic)

    def arrow(self, x1, y1, x2, y2, color=INK, sw=2.0, dash=None, head=8):
        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - head*math.cos(ang), y2 - head*math.sin(ang)
        self.line(x1, y1, bx, by, color, sw, dash)
        a1 = ang + math.radians(150); a2 = ang - math.radians(150)
        p1 = (x2 + head*math.cos(a1), y2 + head*math.sin(a1))
        p2 = (x2 + head*math.cos(a2), y2 + head*math.sin(a2))
        self.parts.append(
            f'<polygon points="{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} '
            f'{p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>')
    def save(self, name):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
               f'height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
               f'font-family="{FONT}">\n'
               f'<rect width="{self.w}" height="{self.h}" fill="white"/>\n'
               f'{body}\n</svg>\n')
        with open(os.path.join(OUT, name), "w") as f:
            f.write(svg)
        print("wrote", name)

# confidence feedback loop
def fig_feedback():
    s = SVG(1000, 440)
    s.text(500, 32, "Model-in-the-loop confidence feedback", 19, weight="bold")
    s.text(500, 53, "spend the next annotations where the model is demonstrably weak",
           12.5, fill=MUTE, italic=True)

    nodes = [
        (BLUE, "Train M", ["on current","labelled set"], 70),
        (TEAL, "Predict", ["full-image","probabilities p"], 270),
        (CRIM, "Uncertainty", ["entropy H(p) /","|logit| map"], 470),
        (PURPLE, "Priority", ["high-entropy +","Frangi prior"], 670),
        (AMBER, "Next seeds", ["place boxes in","weak regions"], 870),
    ]
    y, bw, bh = 130, 150, 100
    for c,t,body,x in nodes:
        s.rect(x-bw/2, y, bw, bh, PANEL, c, 1.8, 10)
        s.text(x, y+26, t, 13.5, weight="bold", fill=c)
        s.textlines(x, y+50, body, 11.5, lh=16)
    for i in range(len(nodes)-1):
        s.arrow(nodes[i][3]+bw/2+2, y+bh/2, nodes[i+1][3]-bw/2-2, y+bh/2, MUTE, 2)
    # loop back seeds to train
    s.arrow(870, y+bh, 870, 330, AMBER, 2.2, dash="6 5")
    s.line(870, 330, 70, 330, AMBER, 2.2, dash="6 5")
    s.arrow(70, 330, 70, y+bh+2, AMBER, 2.2, dash="6 5")
    s.text(470, 348, "retrain on the enlarged, weakness-targeted labelled set",
           12, fill=AMBER, italic=True)

    # entropy-error validation callout
    s.rect(290, 372, 420, 50, "#eef6ff", BLUE, 1.3, 8)
    s.text(500, 392, "Validated proxy:", 11.5, weight="bold", fill=BLUE)
    s.text(500, 409, "entropy-error Pearson r = 0.989 --> uncertainty reliably marks the weak regions",
           11, fill=INK)
    s.save("fig_confidence_feedback.svg")


# granulometry
def fig_granulometry():
    s = SVG(1000, 470)
    s.text(500, 32, "Granulometry for object-size balancing", 19, weight="bold")
    s.text(500, 53, "estimate vessel calibres by successive openings, then seed to flatten the size distribution",
           12.5, fill=MUTE, italic=True)

    # row of openings with growing disks
    s.text(70, 96, "vessel-likeness", 11.5, anchor="start", weight="bold")
    s.text(70, 110, "map  X", 11.5, anchor="start", fill=MUTE)
    radii = [2, 5, 9, 16]
    xs = [70, 270, 470, 670]
    for i,(r,x) in enumerate(zip(radii, xs)):
        s.rect(x, 130, 150, 110, PANEL, PURPLE, 1.5, 8)
        # stylised vessels that survive opening of radius r
        for j,(yy,th) in enumerate([(160,2),(185,5),(210,9),(232,15)]):
            survives = (th/2) >= r
            col = PURPLE if survives else "#d8cfe6"
            s.line(x+18, yy, x+132, yy, col, th)
        s.text(x+75, 124, f"open( X , disk r={r} )", 11, weight="bold", fill=PURPLE)
        if i < len(xs)-1:
            s.arrow(x+152, 185, xs[i+1]-2, 185, MUTE, 2)
    s.text(745, 185, "...", 18, fill=MUTE)

    # arrow down to spectrum
    s.arrow(420, 250, 420, 286, MUTE, 2)
    s.text(440, 272, "residual area at each scale = pattern spectrum",
           11, anchor="start", fill=MUTE, italic=True)

    # spectrum bars
    bx, by, bw, bh = 120, 300, 360, 120
    s.line(bx, by+bh, bx+bw, by+bh, INK, 1.5)
    s.line(bx, by, bx, by+bh, INK, 1.5)
    obs = [0.42, 0.27, 0.16, 0.09, 0.04, 0.02]
    n = len(obs); bwid = bw/(n*1.4)
    for i,v in enumerate(obs):
        x = bx + 12 + i*(bw/n)
        s.rect(x, by+bh-v*bh, bwid, v*bh, PURPLE, PURPLE, 0, 2, opacity=0.85)
    s.line(bx, by+bh-(1.0/n)*bh, bx+bw, by+bh-(1.0/n)*bh, GREEN, 2, dash="6 4")
    s.text(bx+bw, by+bh-(1.0/n)*bh-6, "uniform target", 10.5, anchor="end", fill=GREEN)
    s.text(bx+bw/2, by+bh+22, "vessel calibre  (thin to thick)", 11, fill=MUTE)
    s.text(bx-8, by-4, "frequency", 10.5, anchor="end", fill=MUTE)
    s.text(bx+bw/2, by-8, "observed dataset distribution", 11, weight="bold")

    s.arrow(bx+34, by+10, bx+34, by+40, CRIM, 2.2)
    s.text(bx+44, by+28, "suppress", 10.5, anchor="start", fill=CRIM)
    s.arrow(bx+12+5*(bw/n)+bwid/2, by+bh-10, bx+12+5*(bw/n)+bwid/2, by+bh-44, GREEN, 2.2)
    s.text(bx+12+5*(bw/n)-2, by+bh-50, "enhance", 10.5, anchor="middle", fill=GREEN)

    s.rect(560, 300, 340, 120, PANEL, EDGE, 1.3, 10)
    s.text(580, 324, "Seed-selection weight", 12.5, anchor="start", weight="bold")
    s.text(580, 346, "w(scale) = 1 / ( dataset_freq(scale) + eps )",
           11.5, anchor="start", fill=PURPLE)
    s.textlines(580, 368, ["Boxes rich in rare large-calibre vessels score",
        "far above boxes of common thin branches, so the",
        "annotated set converges toward a uniform",
        "representation of all vessel morphologies."],
        11, anchor="start", lh=15.5, fill=INK)
    s.save("fig_granulometry.svg")


if __name__ == "__main__":
    fig_feedback(); fig_granulometry()
    print("diagrams done", OUT)
