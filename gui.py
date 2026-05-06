import numpy as np
import tkinter as tk
from tkinter import ttk, font
import math, random, time

scores  = np.load("final_candidate_scores.npy")
labels  = np.load("final_candidate_labels.npy")
tiers   = np.load("final_candidate_tiers.npy")
ranking = np.load("final_candidate_ranking.npy")
agree   = np.load("final_agreement_counts.npy")
bay     = np.load("bayesian_candidate_scores.npy")
cnn     = np.load("cnn_candidate_scores.npy")

n_high   = int(np.sum(tiers == "HIGH"))
n_medium = int(np.sum(tiers == "MEDIUM"))
n_low    = int(np.sum(tiers == "LOW"))
n_total  = len(scores)

# ── Softer colour palette ─────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
PANEL2   = "#1c2330"
BORDER   = "#21303f"
ACCENT   = "#79c0ff"   # softer blue
GOLD     = "#e3b341"
GREEN    = "#56d364"
ORANGE   = "#d29922"
RED      = "#f85149"
TEXT     = "#cdd9e5"
SUBTEXT  = "#768390"
HIGH_COL = "#56d364"
MED_COL  = "#e3b341"
LOW_COL  = "#f85149"
DIVERG   = "#d2a679"   # warm peach

def tier_color(t):
    return HIGH_COL if t == "HIGH" else (MED_COL if t == "MEDIUM" else LOW_COL)

def dim(hex_col, factor):
    r = min(255, int(int(hex_col[1:3], 16) * factor))
    g = min(255, int(int(hex_col[3:5], 16) * factor))
    b = min(255, int(int(hex_col[5:7], 16) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Starfield ─────────────────────────────────────────────────────────────────
class Starfield:
    def __init__(self, canvas, w, h, n=55):
        self.canvas   = canvas
        self.stars    = []
        self._running = True
        for _ in range(n):
            x     = random.uniform(0, w)
            y     = random.uniform(0, h)
            r     = random.uniform(0.4, 1.8)
            sp    = random.uniform(0.003, 0.010)
            br    = random.uniform(0.20, 0.65)
            phase = random.uniform(0, math.pi * 2)
            sid   = canvas.create_oval(x-r, y-r, x+r, y+r,
                                       fill="#aabbcc", outline="")
            self.stars.append((sid, sp, br, phase))
        self._animate()

    def _animate(self):
        if not self._running:
            return
        t = time.time()
        for sid, sp, br, phase in self.stars:
            pulse = 0.5 + 0.5 * math.sin(t * sp * 5 + phase)
            alpha = int(190 * br * pulse)
            col   = f"#{alpha:02x}{alpha:02x}{min(255,alpha+25):02x}"
            self.canvas.itemconfig(sid, fill=col)
        self.canvas.after(60, self._animate)

    def stop(self):
        self._running = False


# ── Main application ──────────────────────────────────────────────────────────
class ExoplanetGUI:
    def __init__(self, root):
        self.root = root
        root.title("🔭  Exoplanet Candidate Explorer  |  NASA Kepler KOI Dataset")
        root.configure(bg=BG)
        root.geometry("1340x860")
        root.minsize(1100, 700)

        self._sort_reverse         = {}
        self._scatter_items        = []
        self._scatter_tooltip      = None
        self._scatter_highlight_id = None
        self._scatter_highlight_ids = []

        # Pre-compute data range for scatter auto-zoom
        all_bay  = bay.astype(float)
        all_cnn  = cnn.astype(float)
        margin   = 0.04
        self._sc_xmin = max(0.0, float(all_bay.min()) - margin)
        self._sc_xmax = min(1.0, float(all_bay.max()) + margin)
        self._sc_ymin = max(0.0, float(all_cnn.min()) - margin)
        self._sc_ymax = min(1.0, float(all_cnn.max()) + margin)

        self._build_fonts()
        self._build_layout()
        self._populate_table()
        self.root.after(150, self._draw_scatter)

    # ── Fonts ─────────────────────────────────────────────────────────────────
    def _build_fonts(self):
        self.f_title = font.Font(family="Courier New", size=16, weight="bold")
        self.f_head  = font.Font(family="Courier New", size=10, weight="bold")
        self.f_body  = font.Font(family="Courier New", size=10)
        self.f_small = font.Font(family="Courier New", size=9)
        self.f_stat  = font.Font(family="Courier New", size=21, weight="bold")
        self.f_sub   = font.Font(family="Courier New", size=8)
        self.f_big   = font.Font(family="Courier New", size=26, weight="bold")

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_layout(self):
        hdr = tk.Frame(self.root, bg=PANEL, height=60)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        self.hdr_canvas = tk.Canvas(hdr, bg=PANEL, highlightthickness=0, height=60)
        self.hdr_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self.hdr_canvas.after(200, self._start_starfield)
        tk.Label(hdr, text="🔭  EXOPLANET CANDIDATE EXPLORER",
                 font=self.f_title, bg=PANEL, fg=ACCENT).place(x=22, y=12)
        tk.Label(hdr,
                 text="NASA Kepler KOI  •  Multi-Method AI Pipeline  •  IBA Karachi 2026",
                 font=self.f_sub, bg=PANEL, fg=SUBTEXT).place(x=25, y=38)

        stat_bar = tk.Frame(self.root, bg=BG)
        stat_bar.pack(fill="x", padx=12, pady=(8, 4))
        self._stat_card(stat_bar, str(n_total),  "CANDIDATES",   ACCENT)
        self._stat_card(stat_bar, str(n_high),   "HIGH CONF.",   HIGH_COL)
        self._stat_card(stat_bar, str(n_medium), "MEDIUM CONF.", MED_COL)
        self._stat_card(stat_bar, str(n_low),    "LOW / FP",     LOW_COL)
        self._stat_card(stat_bar, "99.12%",      "BAYESIAN ACC", GOLD)
        self._stat_card(stat_bar, "99.30%",      "CNN ACC",      GREEN)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(body, bg=BG, width=340)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)
        self._build_table(left)
        self._build_right(right)

    def _start_starfield(self):
        w = self.hdr_canvas.winfo_width()
        self.sf = Starfield(self.hdr_canvas, w, 60)

    def _stat_card(self, parent, value, label, color):
        f = tk.Frame(parent, bg=PANEL, bd=0, relief="flat",
                     highlightbackground=BORDER, highlightthickness=1)
        f.pack(side="left", padx=5, pady=4, ipadx=13, ipady=5)
        tk.Label(f, text=value, font=self.f_stat, bg=PANEL, fg=color).pack()
        tk.Label(f, text=label, font=self.f_sub,  bg=PANEL, fg=SUBTEXT).pack()

    # ── Table ─────────────────────────────────────────────────────────────────
    def _build_table(self, parent):
        ctrl = tk.Frame(parent, bg=BG)
        ctrl.pack(fill="x", pady=(0, 6))
        tk.Label(ctrl, text="RANKED CANDIDATES", font=self.f_head,
                 bg=BG, fg=TEXT).pack(side="left")
        tk.Label(ctrl, text="  Search:", font=self.f_body,
                 bg=BG, fg=SUBTEXT).pack(side="left", padx=(20, 4))
        self.search_var = tk.StringVar()
        self.search_var.trace("w", lambda *_: self._filter_table())
        se = tk.Entry(ctrl, textvariable=self.search_var,
                      font=self.f_body, bg=PANEL, fg=TEXT,
                      insertbackground=ACCENT, relief="flat",
                      highlightbackground=BORDER, highlightthickness=1, width=14)
        se.pack(side="left")
        tk.Label(ctrl, text="  Tier:", font=self.f_body,
                 bg=BG, fg=SUBTEXT).pack(side="left", padx=(12, 4))
        self.tier_var = tk.StringVar(value="ALL")
        for t in ["ALL", "HIGH", "MEDIUM", "LOW"]:
            rb = tk.Radiobutton(ctrl, text=t, variable=self.tier_var,
                                value=t, command=self._filter_table,
                                font=self.f_small, bg=BG,
                                fg=tier_color(t) if t != "ALL" else TEXT,
                                selectcolor=PANEL, activebackground=BG,
                                relief="flat")
            rb.pack(side="left", padx=2)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Ex.Treeview",
                        background=PANEL, foreground=TEXT,
                        fieldbackground=PANEL, rowheight=24,
                        font=("Courier New", 9))
        style.configure("Ex.Treeview.Heading",
                        background=PANEL2, foreground=ACCENT,
                        font=("Courier New", 9, "bold"), relief="flat")
        style.map("Ex.Treeview",
                  background=[("selected", "#1f3350")],
                  foreground=[("selected", ACCENT)])

        cols = ("Rank","Index","Score","Tier","Agree","Bayesian","CNN","Δ")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings",
                                 style="Ex.Treeview", selectmode="browse")
        widths = {"Rank":52,"Index":58,"Score":88,"Tier":78,
                  "Agree":65,"Bayesian":88,"CNN":78,"Δ":65}
        for c in cols:
            self.tree.heading(c, text=c, command=lambda _c=c: self._sort_col(_c))
            self.tree.column(c, width=widths[c], anchor="center")
        sb = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        self.tree.tag_configure("HIGH",        foreground=HIGH_COL)
        self.tree.tag_configure("MEDIUM",      foreground=MED_COL)
        self.tree.tag_configure("LOW",         foreground=LOW_COL)
        self.tree.tag_configure("HIGH_alt",    foreground=HIGH_COL,  background="#131a1f")
        self.tree.tag_configure("MEDIUM_alt",  foreground=MED_COL,   background="#131a1f")
        self.tree.tag_configure("LOW_alt",     foreground=LOW_COL,   background="#131a1f")
        self.tree.tag_configure("DIVERGE",     foreground=DIVERG)
        self.tree.tag_configure("DIVERGE_alt", foreground=DIVERG,    background="#131a1f")

    def _populate_table(self):
        self.all_rows = []
        for rank, idx in enumerate(ranking, 1):
            idx   = int(idx)
            delta = abs(float(bay[idx]) - float(cnn[idx]))
            row   = (rank, idx, f"{scores[idx]:.4f}", tiers[idx],
                     f"{int(agree[idx])}/5", f"{bay[idx]:.3f}",
                     f"{cnn[idx]:.3f}", f"{delta:.3f}")
            self.all_rows.append((row, tiers[idx], delta, rank))
        self._fill_tree(self.all_rows)

    def _fill_tree(self, rows):
        self.tree.delete(*self.tree.get_children())
        for i, (row, tier, delta, rank) in enumerate(rows):
            alt = "_alt" if i % 2 == 0 else ""
            tag = ("DIVERGE" + alt) if delta > 0.05 else (tier + alt)
            self.tree.insert("", "end", values=row, tags=(tag,))

    def _filter_table(self):
        q    = self.search_var.get().strip().lower()
        tier = self.tier_var.get()
        filtered = [r for r in self.all_rows
                    if (tier == "ALL" or r[1] == tier)
                    and (not q or any(q in str(v).lower() for v in r[0]))]
        self._fill_tree(filtered)

    def _sort_col(self, col):
        col_map = {"Rank":0,"Index":1,"Score":2,"Tier":3,
                   "Agree":4,"Bayesian":5,"CNN":6,"Δ":7}
        ci  = col_map[col]
        rev = self._sort_reverse.get(col, False)
        self._sort_reverse[col] = not rev
        try:
            self.all_rows.sort(
                key=lambda r: float(str(r[0][ci]).split("/")[0]),
                reverse=not rev)
        except ValueError:
            self.all_rows.sort(key=lambda r: str(r[0][ci]), reverse=not rev)
        self._filter_table()

    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        idx  = int(vals[1])
        self._update_detail(idx, int(vals[0]))
        self._highlight_scatter(idx)

    # ── Right panel ───────────────────────────────────────────────────────────
    def _build_right(self, parent):
        scatter_frame = tk.Frame(parent, bg=PANEL,
                                 highlightbackground=BORDER, highlightthickness=1)
        scatter_frame.pack(fill="x", pady=(0, 8))

        hrow = tk.Frame(scatter_frame, bg=PANEL)
        hrow.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(hrow, text="MODEL AGREEMENT", font=self.f_head,
                 bg=PANEL, fg=TEXT).pack(side="left")
        tk.Label(hrow, text="  Bayesian vs CNN",
                 font=self.f_sub, bg=PANEL, fg=SUBTEXT).pack(side="left")

        legend_row = tk.Frame(scatter_frame, bg=PANEL)
        legend_row.pack(fill="x", padx=10, pady=(0, 4))
        for lbl, col in [("High", HIGH_COL), ("Medium", MED_COL),
                          ("Low/FP", LOW_COL), ("Diverge", DIVERG)]:
            dot = tk.Canvas(legend_row, width=8, height=8, bg=PANEL,
                            highlightthickness=0)
            dot.create_oval(0, 0, 7, 7, fill=col, outline="")
            dot.pack(side="left", padx=(0, 2))
            tk.Label(legend_row, text=lbl, font=self.f_sub,
                     bg=PANEL, fg=SUBTEXT).pack(side="left", padx=(0, 8))

        self.scatter = tk.Canvas(scatter_frame, bg=PANEL, height=215,
                                 highlightthickness=0)
        self.scatter.pack(fill="x", padx=8, pady=(0, 8))
        self.scatter.bind("<Motion>",   self._scatter_hover)
        self.scatter.bind("<Leave>",    self._scatter_leave)
        self.scatter.bind("<Button-1>", self._scatter_click)

        detail_frame = tk.Frame(parent, bg=PANEL,
                                highlightbackground=BORDER, highlightthickness=1)
        detail_frame.pack(fill="both", expand=True)
        tk.Label(detail_frame, text="CANDIDATE DETAIL", font=self.f_head,
                 bg=PANEL, fg=TEXT).pack(pady=(8, 4))
        scroll_wrap = tk.Frame(detail_frame, bg=PANEL)
        scroll_wrap.pack(fill="both", expand=True)
        dsb = ttk.Scrollbar(scroll_wrap, orient="vertical")
        dsb.pack(side="right", fill="y")
        self.detail_canvas = tk.Canvas(scroll_wrap, bg=PANEL,
                                       highlightthickness=0,
                                       yscrollcommand=dsb.set)
        self.detail_canvas.pack(side="left", fill="both", expand=True,
                                padx=10, pady=(0, 10))
        dsb.config(command=self.detail_canvas.yview)
        self._show_placeholder()

    # ── Scatter plot ──────────────────────────────────────────────────────────
    def _draw_scatter(self):
        self.scatter.update_idletasks()
        w   = self.scatter.winfo_width() or 318
        h   = 215
        pl  = 38   # left  (y labels)
        pr  = 10   # right
        pt  = 12   # top
        pb  = 28   # bottom (x labels)
        pw  = w - pl - pr
        ph  = h - pt - pb

        xmin, xmax = self._sc_xmin, self._sc_xmax
        ymin, ymax = self._sc_ymin, self._sc_ymax
        xspan = (xmax - xmin) or 1.0
        yspan = (ymax - ymin) or 1.0

        def to_canvas(bv, cv):
            cx = pl + int((bv - xmin) / xspan * pw)
            cy = pt + ph - int((cv - ymin) / yspan * ph)
            return cx, cy

        self.scatter.delete("all")
        self._scatter_items.clear()
        self._to_canvas_fn = to_canvas

        # ── Subtle grid lines ─────────────────────────────────────────────────
        n_ticks = 4
        for i in range(n_ticks + 1):
            t   = i / n_ticks
            gxv = xmin + t * xspan
            gyv = ymin + t * yspan
            gx  = pl + int(t * pw)
            gy  = pt + ph - int(t * ph)
            self.scatter.create_line(gx, pt, gx, pt+ph,
                                     fill=BORDER, width=1, dash=(2, 5))
            self.scatter.create_line(pl, gy, pl+pw, gy,
                                     fill=BORDER, width=1, dash=(2, 5))
            self.scatter.create_text(gx, pt+ph+9,
                                     text=f"{gxv:.2f}", font=self.f_sub,
                                     fill=SUBTEXT, anchor="n")
            self.scatter.create_text(pl-5, gy,
                                     text=f"{gyv:.2f}", font=self.f_sub,
                                     fill=SUBTEXT, anchor="e")

        # ── Diagonal agreement line (y=x clipped to viewport) ─────────────────
        lo = max(xmin, ymin)
        hi = min(xmax, ymax)
        if lo < hi:
            cx0, cy0 = to_canvas(lo, lo)
            cx1, cy1 = to_canvas(hi, hi)
            self.scatter.create_line(cx0, cy0, cx1, cy1,
                                     fill=dim(ACCENT, 0.28), dash=(5, 4), width=1)

        # ── Soft divergence band shading (|Δ| ≤ 0.05) ────────────────────────
        band  = 0.05
        steps = 40
        top_pts, bot_pts = [], []
        for si in range(steps + 1):
            t  = si / steps
            bv = xmin + t * xspan
            if ymin <= bv + band <= ymax:
                top_pts.extend(to_canvas(bv, bv + band))
            if ymin <= bv - band <= ymax:
                bot_pts.extend(to_canvas(bv, bv - band))
        if len(top_pts) >= 4:
            corners = [pl+pw, pt+ph, pl, pt+ph]
            self.scatter.create_polygon(top_pts + corners,
                                        fill=dim(DIVERG, 0.055), outline="")
        if len(bot_pts) >= 4:
            corners = [pl+pw, pt, pl, pt]
            self.scatter.create_polygon(bot_pts + corners,
                                        fill=dim(DIVERG, 0.055), outline="")

        # ── Axes (drawn after grid so they sit on top) ────────────────────────
        self.scatter.create_line(pl, pt, pl, pt+ph,       fill=dim(BORDER, 1.8), width=1)
        self.scatter.create_line(pl, pt+ph, pl+pw, pt+ph, fill=dim(BORDER, 1.8), width=1)

        # ── Axis labels ───────────────────────────────────────────────────────
        self.scatter.create_text(pl + pw//2, h - 4,
                                 text="Bayesian →",
                                 font=self.f_sub, fill=dim(SUBTEXT, 0.85))
        self.scatter.create_text(10, pt + ph//2,
                                 text="CNN", font=self.f_sub,
                                 fill=dim(SUBTEXT, 0.85), angle=90)

        # ── Dots — draw LOW first, then MEDIUM, HIGH, then DIVERGE on top ─────
        tier_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        indexed = [(i, float(bay[i]), float(cnn[i]),
                    abs(float(bay[i]) - float(cnn[i])), tiers[i])
                   for i in range(n_total)]
        indexed.sort(key=lambda x: (1 if x[3] > 0.05 else 0,
                                    tier_order.get(x[4], 0)))

        for i, bv, cv, delta, tier in indexed:
            cx, cy = to_canvas(bv, cv)
            if not (pl <= cx <= pl+pw and pt <= cy <= pt+ph):
                continue
            diverge = delta > 0.05
            col = DIVERG if diverge else dim(tier_color(tier), 0.80)
            r   = 3 if diverge else 2
            sid = self.scatter.create_oval(
                cx-r, cy-r, cx+r, cy+r, fill=col, outline="")
            self._scatter_items.append((sid, i, cx, cy, r))

        self._scatter_highlight_id  = None
        self._scatter_highlight_ids = []
        self._scatter_meta = dict(pl=pl, pr=pr, pt=pt, pb=pb,
                                  pw=pw, ph=ph, w=w, h=h)

    # ── Scatter interactions ──────────────────────────────────────────────────
    def _scatter_hover(self, event):
        hit = self._scatter_hit_test(event.x, event.y)
        if hit is None:
            self._scatter_leave(event)
            return
        _, idx, xp, yp, _ = hit
        if self._scatter_tooltip:
            self.scatter.delete("tooltip")
        m     = self._scatter_meta
        delta = abs(float(bay[idx]) - float(cnn[idx]))
        tip   = (f"#{idx}  {tiers[idx]}\n"
                 f"Bay: {bay[idx]:.4f}\n"
                 f"CNN: {cnn[idx]:.4f}\n"
                 f"Δ:   {delta:.4f}")
        right_half = xp > m["pl"] + m["pw"] * 0.6
        tx = xp - 8 if right_half else xp + 8
        ta = "se"   if right_half else "sw"
        tid = self.scatter.create_text(tx, yp - 4, text=tip,
                                       font=self.f_sub, fill=TEXT,
                                       anchor=ta, tags=("tooltip",))
        bbox = self.scatter.bbox(tid)
        if bbox:
            bg = self.scatter.create_rectangle(
                bbox[0]-4, bbox[1]-3, bbox[2]+4, bbox[3]+3,
                fill=PANEL2, outline=dim(BORDER, 1.4), tags=("tooltip",))
            self.scatter.tag_raise(tid)
        self._scatter_tooltip = "tooltip"
        self.scatter.config(cursor="hand2")

    def _scatter_leave(self, event):
        if self._scatter_tooltip:
            self.scatter.delete("tooltip")
            self._scatter_tooltip = None
        self.scatter.config(cursor="")

    def _scatter_click(self, event):
        hit = self._scatter_hit_test(event.x, event.y)
        if hit is None:
            return
        _, idx, _, _, _ = hit
        where    = np.where(ranking == idx)[0]
        rank_pos = int(where[0]) + 1 if len(where) else 0
        self._update_detail(idx, rank_pos)
        self._highlight_scatter(idx)

    def _scatter_hit_test(self, mx, my, radius=7):
        for sid, idx, xp, yp, r in reversed(self._scatter_items):
            if abs(mx - xp) <= radius and abs(my - yp) <= radius:
                return (sid, idx, xp, yp, r)
        return None

    def _highlight_scatter(self, target_idx):
        for hid in self._scatter_highlight_ids:
            self.scatter.delete(hid)
        self._scatter_highlight_ids = []
        for sid, idx, xp, yp, r in self._scatter_items:
            if idx == target_idx:
                h2 = self.scatter.create_oval(
                    xp-7, yp-7, xp+7, yp+7,
                    fill="", outline=dim(ACCENT, 0.4), width=1)
                h1 = self.scatter.create_oval(
                    xp-5, yp-5, xp+5, yp+5,
                    fill="", outline=ACCENT, width=2)
                self.scatter.tag_raise(h2)
                self.scatter.tag_raise(h1)
                self._scatter_highlight_ids = [h1, h2]
                break

    # ── Detail panel ──────────────────────────────────────────────────────────
    def _show_placeholder(self):
        self.detail_canvas.delete("all")
        self.detail_canvas.create_text(
            145, 80,
            text="← Select a candidate\n   to view details",
            font=self.f_body, fill=SUBTEXT, justify="center")

    def _hline(self, c, y, w):
        c.create_line(10, y, w-10, y, fill=BORDER)
        return y + 10

    def _update_detail(self, idx, rank):
        c     = self.detail_canvas
        c.delete("all")
        c.update_idletasks()
        w     = c.winfo_width() or 310
        score = float(scores[idx])
        tier  = tiers[idx]
        t_col = tier_color(tier)
        ag    = int(agree[idx])
        bay_s = float(bay[idx])
        cnn_s = float(cnn[idx])
        delta = abs(bay_s - cnn_s)
        y     = 14

        # ID + rank
        c.create_text(w//2, y, text=f"Candidate #{idx}",
                      font=self.f_head, fill=ACCENT)
        y += 18
        c.create_text(w//2, y, text=f"Rank #{rank} of {n_total}",
                      font=self.f_small, fill=SUBTEXT)
        y += 26

        # Big score
        c.create_text(w//2, y+16, text=f"{score:.4f}", font=self.f_big, fill=t_col)
        y += 36
        c.create_text(w//2, y, text="ENSEMBLE SCORE", font=self.f_sub, fill=SUBTEXT)
        y += 18

        # Tier badge
        bx0, bx1 = w//2 - 42, w//2 + 42
        c.create_rectangle(bx0, y, bx1, y+17,
                           fill=dim(t_col, 0.14), outline=dim(t_col, 0.55), width=1)
        c.create_text(w//2, y+8, text=tier, font=self.f_small, fill=t_col)
        y += 26

        # Divergence warning
        if delta > 0.05:
            severity = "HIGH" if delta > 0.2 else "MODERATE"
            c.create_rectangle(8, y, w-8, y+32,
                               fill=dim(DIVERG, 0.07),
                               outline=dim(DIVERG, 0.45), width=1)
            c.create_text(w//2, y+9,
                          text=f"⚠  {severity} MODEL DISAGREEMENT",
                          font=self.f_small, fill=DIVERG)
            c.create_text(w//2, y+22,
                          text=f"Δ = {delta:.4f}  —  manual review recommended",
                          font=self.f_sub, fill=dim(DIVERG, 0.8))
            y += 42

        y += 4
        y = self._hline(c, y, w)

        # Classifier agreement
        c.create_text(10, y, text="Classifiers agreed:",
                      font=self.f_small, fill=SUBTEXT, anchor="w")
        c.create_text(w-10, y, text=f"{ag}/5",
                      font=self.f_small, fill=t_col, anchor="e")
        y += 18
        cl_labels = ["DT", "NB", "KM", "Bay", "CNN"]
        dot_r     = 8
        spacing   = 44
        x0        = w//2 - (len(cl_labels)-1) * spacing // 2
        for i, lb in enumerate(cl_labels):
            xc  = x0 + i * spacing
            col = HIGH_COL if i < ag else dim(BORDER, 1.5)
            if i < ag:
                c.create_oval(xc-dot_r-2, y-2, xc+dot_r+2, y+dot_r*2+2,
                              fill=dim(HIGH_COL, 0.07), outline="")
            c.create_oval(xc-dot_r, y, xc+dot_r, y+dot_r*2,
                          fill=col, outline=dim(col, 0.55) if i < ag else "")
            c.create_text(xc, y+dot_r*2+7, text=lb, font=self.f_sub,
                          fill=HIGH_COL if i < ag else SUBTEXT)
        y += dot_r*2 + 24

        y = self._hline(c, y, w)

        # Score bars
        bw = w - 20
        for label, val, col in [("Bayesian", bay_s, ACCENT),
                                 ("CNN",      cnn_s, GREEN)]:
            c.create_text(10, y, text=label, font=self.f_small,
                          fill=SUBTEXT, anchor="w")
            c.create_text(w-10, y, text=f"{val:.4f}",
                          font=self.f_small, fill=col, anchor="e")
            y += 14
            filled = max(1, int(val * bw))
            c.create_rectangle(10, y, 10+bw, y+7,
                               fill=dim(BORDER, 0.65), outline="")
            c.create_rectangle(10, y, 10+filled, y+7, fill=col, outline="")
            y += 18

        y += 4
        y = self._hline(c, y, w)

        # Mini bar chart
        c.create_text(10, y, text="Score comparison",
                      font=self.f_small, fill=SUBTEXT, anchor="w")
        y += 14
        chart_h = 52
        chart_w = w - 20
        c.create_rectangle(10, y, 10+chart_w, y+chart_h,
                           fill=dim(PANEL, 0.35), outline=BORDER)
        bar_w = chart_w // 3
        for bi, (lbl, val, col) in enumerate([
                ("Ens.", score, t_col),
                ("Bay.", bay_s, ACCENT),
                ("CNN",  cnn_s, GREEN)]):
            bx  = 10 + bi*bar_w + 8
            bww = bar_w - 16
            bfh = max(2, int(val * (chart_h - 16)))
            by0 = y + chart_h - 4 - bfh
            c.create_rectangle(bx, by0, bx+bww, y+chart_h-4,
                               fill=col, outline="")
            c.create_text(bx+bww//2, y+chart_h+1,
                          text=lbl, font=self.f_sub, fill=SUBTEXT, anchor="s")
            c.create_text(bx+bww//2, by0-2,
                          text=f"{val:.3f}", font=self.f_sub, fill=col, anchor="s")
        y += chart_h + 18

        c.config(scrollregion=(0, 0, w, y + 10))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = ExoplanetGUI(root)
    root.mainloop()