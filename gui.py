import numpy as np
import tkinter as tk
from tkinter import ttk, font
import math, random, time, threading


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

# Colour palette 
BG        = "#04060f"
PANEL     = "#080d1a"
BORDER    = "#1a2540"
ACCENT    = "#4fc3f7"
GOLD      = "#ffd54f"
GREEN     = "#69f0ae"
ORANGE    = "#ff9800"
RED       = "#ef5350"
TEXT      = "#e8eaf6"
SUBTEXT   = "#7986cb"
HIGH_COL  = "#69f0ae"
MED_COL   = "#ffd54f"
LOW_COL   = "#ef5350"

def tier_color(t):
    return HIGH_COL if t == "HIGH" else (MED_COL if t == "MEDIUM" else LOW_COL)

# STARFIELD CANVAS

class Starfield:
    def __init__(self, canvas, w, h, n=180):
        self.canvas = canvas
        self.w, self.h = w, h
        self.stars = []
        for _ in range(n):
            x  = random.uniform(0, w)
            y  = random.uniform(0, h)
            r  = random.uniform(0.4, 2.2)
            sp = random.uniform(0.003, 0.012)
            br = random.uniform(0.3, 1.0)
            self.stars.append([x, y, r, sp, br, random.uniform(0, math.pi*2)])
        self._ids = []
        self._running = True
        self._animate()

    def _animate(self):
        if not self._running:
            return
        c = self.canvas
        for sid in self._ids:
            c.delete(sid)
        self._ids.clear()
        t = time.time()
        for s in self.stars:
            pulse = 0.55 + 0.45 * math.sin(t * s[3] * 6 + s[5])
            alpha = int(255 * s[4] * pulse)
            col = f"#{alpha:02x}{alpha:02x}{min(255,alpha+40):02x}"
            x, y, r = s[0], s[1], s[2]
            sid = c.create_oval(x-r, y-r, x+r, y+r, fill=col, outline="")
            self._ids.append(sid)
        c.after(50, self._animate)

    def stop(self):
        self._running = False


# MAIN APPLICATION
class ExoplanetGUI:
    def __init__(self, root):
        self.root = root
        root.title("🔭  Exoplanet Candidate Explorer  |  NASA Kepler KOI Dataset")
        root.configure(bg=BG)
        root.geometry("1280x820")
        root.minsize(1000, 680)

        self._build_fonts()
        self._build_layout()
        self._populate_table()
        self._draw_chart()

    # fonts
    def _build_fonts(self):
        self.f_title  = font.Font(family="Courier New", size=17, weight="bold")
        self.f_head   = font.Font(family="Courier New", size=11, weight="bold")
        self.f_body   = font.Font(family="Courier New", size=10)
        self.f_small  = font.Font(family="Courier New", size=9)
        self.f_stat   = font.Font(family="Courier New", size=22, weight="bold")
        self.f_sub    = font.Font(family="Courier New", size=9)

    # top-level layout
    def _build_layout(self):
        # header
        hdr = tk.Frame(self.root, bg=PANEL, height=64)
        hdr.pack(fill="x", padx=0, pady=0)
        hdr.pack_propagate(False)

        # starfield behind header
        self.hdr_canvas = tk.Canvas(hdr, bg=PANEL, highlightthickness=0, height=64)
        self.hdr_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self.hdr_canvas.after(200, lambda: self._start_starfield())

        tk.Label(hdr, text="🔭  EXOPLANET CANDIDATE EXPLORER",
                 font=self.f_title, bg=PANEL, fg=ACCENT).place(x=24, y=18)
        tk.Label(hdr, text="NASA Kepler KOI  •  Multi-Method AI Pipeline  •  IBA Karachi 2026",
                 font=self.f_small, bg=PANEL, fg=SUBTEXT).place(x=27, y=42)

        # stat bar
        stat_bar = tk.Frame(self.root, bg=BG)
        stat_bar.pack(fill="x", padx=12, pady=(10, 4))
        self._stat_card(stat_bar, str(n_total), "CANDIDATES", ACCENT)
        self._stat_card(stat_bar, str(n_high),  "HIGH CONF.",  HIGH_COL)
        self._stat_card(stat_bar, str(n_medium),"MEDIUM CONF.",MED_COL)
        self._stat_card(stat_bar, str(n_low),   "LOW / FP",    LOW_COL)
        self._stat_card(stat_bar, "99.12%",     "BAYESIAN ACC",GOLD)
        self._stat_card(stat_bar, "99.30%",     "CNN ACC",     GREEN)

        # body: left table + right panels
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        left  = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(body, bg=BG, width=310)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        self._build_table(left)
        self._build_right(right)

    def _start_starfield(self):
        w = self.hdr_canvas.winfo_width()
        self.sf = Starfield(self.hdr_canvas, w, 64, n=60)

    def _stat_card(self, parent, value, label, color):
        f = tk.Frame(parent, bg=PANEL, bd=0, relief="flat",
                     highlightbackground=BORDER, highlightthickness=1)
        f.pack(side="left", padx=5, pady=4, ipadx=14, ipady=6)
        tk.Label(f, text=value, font=self.f_stat, bg=PANEL, fg=color).pack()
        tk.Label(f, text=label, font=self.f_sub,  bg=PANEL, fg=SUBTEXT).pack()

    # candidate table
    def _build_table(self, parent):
        ctrl = tk.Frame(parent, bg=BG)
        ctrl.pack(fill="x", pady=(0, 6))

        tk.Label(ctrl, text="RANKED CANDIDATES", font=self.f_head,
                 bg=BG, fg=TEXT).pack(side="left")

        # search
        tk.Label(ctrl, text="  Search:", font=self.f_body,
                 bg=BG, fg=SUBTEXT).pack(side="left", padx=(20, 4))
        self.search_var = tk.StringVar()
        self.search_var.trace("w", lambda *_: self._filter_table())
        se = tk.Entry(ctrl, textvariable=self.search_var,
                      font=self.f_body, bg=PANEL, fg=TEXT,
                      insertbackground=ACCENT, relief="flat",
                      highlightbackground=BORDER, highlightthickness=1, width=14)
        se.pack(side="left")

        # tier filter
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

        # treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Ex.Treeview",
                        background=PANEL, foreground=TEXT,
                        fieldbackground=PANEL, rowheight=24,
                        font=("Courier New", 9))
        style.configure("Ex.Treeview.Heading",
                        background=BORDER, foreground=ACCENT,
                        font=("Courier New", 9, "bold"), relief="flat")
        style.map("Ex.Treeview",
                  background=[("selected", "#1a2e50")],
                  foreground=[("selected", ACCENT)])

        cols = ("Rank", "Index", "Score", "Tier", "Agree", "Bayesian", "CNN")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings",
                                 style="Ex.Treeview", selectmode="browse")

        widths = {"Rank":55,"Index":60,"Score":90,"Tier":80,"Agree":70,"Bayesian":90,"CNN":80}
        for c in cols:
            self.tree.heading(c, text=c,
                              command=lambda _c=c: self._sort_col(_c))
            self.tree.column(c, width=widths[c], anchor="center")

        sb = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # tag colours per tier
        self.tree.tag_configure("HIGH",   foreground=HIGH_COL)
        self.tree.tag_configure("MEDIUM", foreground=MED_COL)
        self.tree.tag_configure("LOW",    foreground=LOW_COL)

    def _populate_table(self):
        self.all_rows = []
        for rank, idx in enumerate(ranking, 1):
            idx = int(idx)
            row = (
                rank,
                idx,
                f"{scores[idx]:.4f}",
                tiers[idx],
                f"{int(agree[idx])}/5",
                f"{bay[idx]:.3f}",
                f"{cnn[idx]:.3f}",
            )
            self.all_rows.append((row, tiers[idx]))

        self._fill_tree(self.all_rows)

    def _fill_tree(self, rows):
        self.tree.delete(*self.tree.get_children())
        for row, tier in rows:
            self.tree.insert("", "end", values=row, tags=(tier,))

    def _filter_table(self):
        q    = self.search_var.get().strip().lower()
        tier = self.tier_var.get()
        filtered = []
        for row, t in self.all_rows:
            if tier != "ALL" and t != tier:
                continue
            if q and not any(q in str(v).lower() for v in row):
                continue
            filtered.append((row, t))
        self._fill_tree(filtered)

    def _sort_col(self, col):
        col_map = {"Rank":0,"Index":1,"Score":2,"Tier":3,"Agree":4,"Bayesian":5,"CNN":6}
        ci = col_map[col]
        try:
            self.all_rows.sort(key=lambda r: float(str(r[0][ci]).split("/")[0]),
                               reverse=True)
        except ValueError:
            self.all_rows.sort(key=lambda r: str(r[0][ci]))
        self._filter_table()

    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        idx  = int(vals[1])
        self._update_detail(idx, int(vals[0]))

    # right panel
    def _build_right(self, parent):
        # tier chart
        chart_frame = tk.Frame(parent, bg=PANEL,
                               highlightbackground=BORDER, highlightthickness=1)
        chart_frame.pack(fill="x", pady=(0, 8))
        tk.Label(chart_frame, text="CONFIDENCE DISTRIBUTION",
                 font=self.f_head, bg=PANEL, fg=TEXT).pack(pady=(8, 4))
        self.chart = tk.Canvas(chart_frame, bg=PANEL, height=140,
                               highlightthickness=0)
        self.chart.pack(fill="x", padx=10, pady=(0, 10))

        # detail panel
        detail_frame = tk.Frame(parent, bg=PANEL,
                                highlightbackground=BORDER, highlightthickness=1)
        detail_frame.pack(fill="both", expand=True)
        tk.Label(detail_frame, text="CANDIDATE DETAIL",
                 font=self.f_head, bg=PANEL, fg=TEXT).pack(pady=(8, 4))

        self.detail_canvas = tk.Canvas(detail_frame, bg=PANEL,
                                       highlightthickness=0)
        self.detail_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._show_placeholder()

    def _draw_chart(self):
        self.chart.update_idletasks()
        w = self.chart.winfo_width() or 280
        h = 130
        self.chart.config(height=h)

        data   = [("HIGH", n_high, HIGH_COL),
                  ("MED",  n_medium, MED_COL),
                  ("LOW",  n_low,   LOW_COL)]
        total  = n_total
        bar_w  = (w - 60) / 3
        max_h  = h - 50

        for i, (label, val, col) in enumerate(data):
            bh  = max(4, int((val / total) * max_h))
            x0  = 10 + i * (bar_w + 10)
            x1  = x0 + bar_w
            y0  = h - 30 - bh
            y1  = h - 30

            # glow effect
            for g in range(4, 0, -1):
                gc = self._dim(col, 0.15 * g)
                self.chart.create_rectangle(x0-g, y0-g, x1+g, y1+g,
                                            fill=gc, outline="")
            self.chart.create_rectangle(x0, y0, x1, y1, fill=col, outline="")

            self.chart.create_text((x0+x1)/2, y0-8, text=str(val),
                                   font=self.f_small, fill=col)
            self.chart.create_text((x0+x1)/2, h-14, text=label,
                                   font=self.f_small, fill=SUBTEXT)

    def _dim(self, hex_col, factor):
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _show_placeholder(self):
        self.detail_canvas.delete("all")
        self.detail_canvas.create_text(
            145, 80,
            text="← Select a candidate\n   to view details",
            font=self.f_body, fill=SUBTEXT, justify="center")

    def _update_detail(self, idx, rank):
        c = self.detail_canvas
        c.delete("all")
        c.update_idletasks()
        w = c.winfo_width() or 280

        score   = scores[idx]
        tier    = tiers[idx]
        t_col   = tier_color(tier)
        ag      = int(agree[idx])
        bay_s   = bay[idx]
        cnn_s   = cnn[idx]

        y = 14
        c.create_text(w//2, y, text=f"Candidate #{idx}",
                      font=self.f_head, fill=ACCENT)
        y += 20
        c.create_text(w//2, y, text=f"Rank #{rank} of {n_total}",
                      font=self.f_small, fill=SUBTEXT)
        y += 22

        # big score
        c.create_text(w//2, y+18, text=f"{score:.4f}",
                      font=self.f_stat, fill=t_col)
        y += 38
        c.create_text(w//2, y, text="ENSEMBLE SCORE",
                      font=self.f_sub, fill=SUBTEXT)
        y += 20

        # tier badge
        c.create_rectangle(w//2-40, y, w//2+40, y+18,
                            fill=self._dim(t_col, 0.2), outline=t_col)
        c.create_text(w//2, y+9, text=tier, font=self.f_small, fill=t_col)
        y += 30

        # agreement
        c.create_text(10, y, text=f"Classifiers agreed:", font=self.f_small,
                      fill=SUBTEXT, anchor="w")
        y += 16
        dot_colors = [GREEN if i < ag else BORDER for i in range(5)]
        for i, dc in enumerate(dot_colors):
            x = 14 + i * 22
            c.create_oval(x, y, x+14, y+14, fill=dc, outline="")
        labels_cl = ["DT","NB","KM","Bay","CNN"]
        for i, lb in enumerate(labels_cl):
            x = 14 + i * 22
            c.create_text(x+7, y+22, text=lb, font=self.f_sub,
                          fill=GREEN if i < ag else SUBTEXT)
        y += 38

        # score bars
        rows = [("Bayesian", bay_s, ACCENT),
                ("CNN",      cnn_s, GREEN)]
        for label, val, col in rows:
            c.create_text(10, y, text=label, font=self.f_small,
                          fill=SUBTEXT, anchor="w")
            c.create_text(w-10, y, text=f"{val:.3f}", font=self.f_small,
                          fill=col, anchor="e")
            y += 14
            bw = w - 20
            bh_full = max(1, int(val * bw))
            c.create_rectangle(10, y, 10+bw, y+8, fill=BORDER, outline="")
            if bh_full > 0:
                for g in range(3, 0, -1):
                    gc = self._dim(col, 0.2 * g)
                    c.create_rectangle(10, y, 10+bh_full, y+8+g,
                                       fill=gc, outline="")
                c.create_rectangle(10, y, 10+bh_full, y+8, fill=col, outline="")
            y += 16


#  ENTRY POINT
if __name__ == "__main__":
    root = tk.Tk()
    app  = ExoplanetGUI(root)
    root.mainloop()
