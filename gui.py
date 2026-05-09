"""
gui.py — Acoustic Levitation Measurement System GUI

Launch:
    python gui.py
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

PYTHON = sys.executable
ROOT   = Path(__file__).resolve().parent


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Acoustic Levitation — Measurement Pipeline")
        self.geometry("860x720")
        self.minsize(700, 560)
        self._build_paths()
        self._build_notebook()
        self._build_log()

    # ── Common paths bar ──────────────────────────────────────────────────────

    def _build_paths(self):
        pf = ttk.LabelFrame(self, text="Common Paths", padding=6)
        pf.pack(fill="x", padx=8, pady=(8, 2))

        self.v_session = self._path_row(pf, "Session dir",     0, kind="dir")
        self.v_box     = self._path_row(pf, "Box config",      1, default="config/box.yaml")
        self.v_cams    = self._path_row(pf, "Cameras config",  2, default="config/cameras.yaml")
        self.v_calib   = self._path_row(pf, "Calibration dir", 3, kind="dir", default="calibration")
        self.v_sim     = self._path_row(pf, "Sim output",      4)

    def _path_row(self, parent, label, row, kind="file", default=""):
        ttk.Label(parent, text=label + ":").grid(row=row, column=0, sticky="w",
                                                  padx=(0, 6), pady=2)
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=52).grid(row=row, column=1,
                                                            sticky="ew", pady=2)
        if kind == "dir":
            cmd = lambda v=var: v.set(filedialog.askdirectory(initialdir=ROOT))
        else:
            cmd = lambda v=var: v.set(filedialog.askopenfilename(initialdir=ROOT))
        ttk.Button(parent, text="…", width=3, command=cmd).grid(row=row, column=2,
                                                                  padx=(4, 0))
        parent.columnconfigure(1, weight=1)
        return var

    # ── Notebook tabs ─────────────────────────────────────────────────────────

    def _build_notebook(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=4)
        self._tab_calibrate()
        self._tab_capture()
        self._tab_extrinsic()
        self._tab_detect()
        self._tab_triangulate()
        self._tab_error_prop()
        self._tab_compare()
        self._tab_full_pipeline()

    # ── Tab 1: Intrinsic calibration ──────────────────────────────────────────

    def _tab_calibrate(self):
        f = ttk.Frame(self.nb, padding=12)
        self.nb.add(f, text="1 · Calibrate")

        self.cal_id      = self._field(f, 0, "Camera ID",         "cam_front")
        self.cal_imgs    = self._browse(f, 1, "Images dir",        kind="dir")
        self.cal_out     = self._field(f, 2, "Output YAML",        "calibration/cam_front_intrinsics.yaml")
        self.cal_sq_x    = self._field(f, 3, "Squares X",          "9")
        self.cal_sq_y    = self._field(f, 4, "Squares Y",          "6")
        self.cal_sq_len  = self._field(f, 5, "Square length (m)",  "0.04")
        self.cal_mk_len  = self._field(f, 6, "Marker length (m)",  "0.02")
        self.cal_dict    = self._field(f, 7, "ArUco dict",         "DICT_5X5_100")
        self.cal_reproj  = self._field(f, 8, "Max reproj (px)",    "1.0")

        ttk.Button(f, text="▶  Run Calibration",
                   command=self._run_calibrate).grid(row=9, column=0, columnspan=3,
                                                     pady=14, ipadx=10, ipady=4)

    # ── Tab 2: Capture ────────────────────────────────────────────────────────

    def _tab_capture(self):
        f = ttk.Frame(self.nb, padding=12)
        self.nb.add(f, text="2 · Capture")

        self.cap_nframes = self._field(f, 0, "Frames per camera", "200")

        bf = ttk.Frame(f)
        bf.grid(row=1, column=0, columnspan=3, pady=14)
        ttk.Button(bf, text="List Cameras",
                   command=self._run_list_cameras).pack(side="left", padx=6,
                                                        ipadx=8, ipady=4)
        ttk.Button(bf, text="▶  Capture Session",
                   command=self._run_capture).pack(side="left", padx=6,
                                                   ipadx=8, ipady=4)

    # ── Tab 3: Extrinsic solver ───────────────────────────────────────────────

    def _tab_extrinsic(self):
        f = ttk.Frame(self.nb, padding=12)
        self.nb.add(f, text="3 · Extrinsic")

        self.ext_min_markers = self._field(f, 0, "Min markers",    "3")
        self.ext_max_reproj  = self._field(f, 1, "Max reproj (px)","2.0")

        ttk.Button(f, text="▶  Run Extrinsic Solver",
                   command=self._run_extrinsic).grid(row=2, column=0, columnspan=3,
                                                     pady=14, ipadx=10, ipady=4)

    # ── Tab 4: Ball detector ──────────────────────────────────────────────────

    def _tab_detect(self):
        f = ttk.Frame(self.nb, padding=12)
        self.nb.add(f, text="4 · Ball Detect")

        self.det_min_area  = self._field(f, 0, "Min blob area (px²)", "50")
        self.det_max_area  = self._field(f, 1, "Max blob area (px²)", "50000")
        self.det_residual  = self._field(f, 2, "Max fit residual (px)", "3.0")
        self.det_roi_r     = self._field(f, 3, "ROI radius (px)",      "60")

        self.det_interactive = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Interactive blob selection",
                        variable=self.det_interactive).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))

        ttk.Label(f, text="Opens a window per camera. All blobs shown numbered.\n"
                          "Click blob or press 1–9 to select. Enter = confirm. ESC = cancel.",
                  foreground="gray", justify="left").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(2, 0))

        ttk.Button(f, text="▶  Run Ball Detector",
                   command=self._run_detect).grid(row=6, column=0, columnspan=3,
                                                  pady=14, ipadx=10, ipady=4)

    # ── Tab 5: Triangulation ──────────────────────────────────────────────────

    def _tab_triangulate(self):
        f = ttk.Frame(self.nb, padding=12)
        self.nb.add(f, text="5 · Triangulate")

        ttk.Label(f, text="Reads session/extrinsics.json and session/ball_detections.json.",
                  foreground="gray").grid(row=0, column=0, columnspan=3,
                                         sticky="w", pady=(0, 12))
        ttk.Button(f, text="▶  Run Triangulation",
                   command=self._run_triangulate).grid(row=1, column=0, columnspan=3,
                                                       ipadx=10, ipady=4)

    # ── Tab 6: Error propagation ──────────────────────────────────────────────

    def _tab_error_prop(self):
        f = ttk.Frame(self.nb, padding=12)
        self.nb.add(f, text="6 · Error Prop")

        self.err_nmc = self._field(f, 0, "Monte Carlo trials", "500")

        ttk.Button(f, text="▶  Run Error Propagation",
                   command=self._run_error_prop).grid(row=1, column=0, columnspan=3,
                                                      pady=14, ipadx=10, ipady=4)

    # ── Tab 7: Comparison ────────────────────────────────────────────────────

    def _tab_compare(self):
        f = ttk.Frame(self.nb, padding=12)
        self.nb.add(f, text="7 · Compare")

        self.cmp_thresh = self._field(f, 0, "Threshold (mm)",       "2.0")
        self.cmp_rank   = self._field(f, 1, "Sim candidate rank",   "1")

        ttk.Button(f, text="▶  Run Comparison",
                   command=self._run_compare).grid(row=2, column=0, columnspan=3,
                                                   pady=14, ipadx=10, ipady=4)

    # ── Tab 8: Full pipeline ──────────────────────────────────────────────────

    def _tab_full_pipeline(self):
        f = ttk.Frame(self.nb, padding=12)
        self.nb.add(f, text="★  Full Pipeline")

        self.fp_thresh      = self._field(f, 0, "Threshold (mm)",      "2.0")
        self.fp_rank        = self._field(f, 1, "Sim candidate rank",  "1")
        self.fp_min_markers = self._field(f, 2, "Min markers",         "3")
        self.fp_max_reproj  = self._field(f, 3, "Max reproj (px)",     "2.0")
        self.fp_min_area    = self._field(f, 4, "Min blob area (px²)", "50")
        self.fp_max_area    = self._field(f, 5, "Max blob area (px²)", "50000")
        self.fp_nmc         = self._field(f, 6, "MC trials",           "500")

        self.fp_skip_err    = tk.BooleanVar(value=False)
        self.fp_interactive = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Skip error propagation (faster)",
                        variable=self.fp_skip_err).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(f, text="Interactive blob selection",
                        variable=self.fp_interactive).grid(
            row=8, column=0, columnspan=3, sticky="w")

        ttk.Button(f, text="▶▶  Run Full Pipeline",
                   command=self._run_full_pipeline).grid(
            row=9, column=0, columnspan=3, pady=18, ipadx=24, ipady=8)

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _field(self, parent, row, label, default=""):
        ttk.Label(parent, text=label + ":").grid(row=row, column=0, sticky="w",
                                                  padx=(0, 8), pady=3)
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=28).grid(row=row, column=1,
                                                            sticky="w", pady=3)
        return var

    def _browse(self, parent, row, label, kind="file"):
        ttk.Label(parent, text=label + ":").grid(row=row, column=0, sticky="w",
                                                  padx=(0, 8), pady=3)
        var = tk.StringVar()
        ttk.Entry(parent, textvariable=var, width=28).grid(row=row, column=1,
                                                            sticky="w", pady=3)
        if kind == "dir":
            cmd = lambda v=var: v.set(filedialog.askdirectory(initialdir=ROOT))
        else:
            cmd = lambda v=var: v.set(filedialog.askopenfilename(initialdir=ROOT))
        ttk.Button(parent, text="…", width=3, command=cmd).grid(row=row, column=2,
                                                                  padx=(4, 0))
        return var

    # ── Log area ──────────────────────────────────────────────────────────────

    def _build_log(self):
        lf = ttk.LabelFrame(self, text="Output Log", padding=4)
        lf.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        self.log = scrolledtext.ScrolledText(
            lf, height=10, state="disabled",
            font=("Consolas", 9),
            background="#1e1e1e", foreground="#d4d4d4",
            insertbackground="white",
        )
        self.log.pack(fill="both", expand=True)
        self.log.tag_config("hdr", foreground="#9cdcfe")
        self.log.tag_config("ok",  foreground="#b5cea8")
        self.log.tag_config("err", foreground="#f48771")
        self.log.tag_config("warn",foreground="#dcdcaa")

        ttk.Button(lf, text="Clear log",
                   command=self._clear_log).pack(anchor="e", pady=(4, 0))

    def _log(self, text: str, tag: str = "") -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text, tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ── Command runner (non-blocking) ─────────────────────────────────────────

    def _check_session(self) -> bool:
        if not self.v_session.get().strip():
            messagebox.showerror("Missing path", "Set Session dir first.")
            return False
        return True

    def _run_command(self, cmd: list[str], header: str) -> None:
        self._log(f"\n{'─' * 58}\n  {header}\n{'─' * 58}\n", "hdr")

        def _worker():
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(ROOT),
                    bufsize=1,
                )
                for line in proc.stdout:
                    lo = line.lower()
                    if any(w in lo for w in ("error", "fail", "traceback")):
                        tag = "err"
                    elif any(w in lo for w in ("warn",)):
                        tag = "warn"
                    elif any(w in lo for w in ("pass", "saved", "done", "complete", "accepted")):
                        tag = "ok"
                    else:
                        tag = ""
                    self.after(0, self._log, line, tag)
                proc.wait()
                result = f"\n[Exit {proc.returncode}]\n"
                self.after(0, self._log, result,
                           "ok" if proc.returncode == 0 else "err")
            except Exception as exc:
                self.after(0, self._log, f"LAUNCH ERROR: {exc}\n", "err")

        threading.Thread(target=_worker, daemon=True).start()

    # ── Step runners ─────────────────────────────────────────────────────────

    def _run_calibrate(self):
        self._run_command([
            PYTHON, "-m", "intrinsic_calibration.calibrate",
            "--camera-id",     self.cal_id.get(),
            "--images-dir",    self.cal_imgs.get(),
            "--output",        self.cal_out.get(),
            "--squares-x",     self.cal_sq_x.get(),
            "--squares-y",     self.cal_sq_y.get(),
            "--square-length", self.cal_sq_len.get(),
            "--marker-length", self.cal_mk_len.get(),
            "--dict",          self.cal_dict.get(),
            "--max-reproj-px", self.cal_reproj.get(),
        ], "Intrinsic Calibration")

    def _run_list_cameras(self):
        self._run_command(
            [PYTHON, "-m", "capture.capture", "--list-cameras"],
            "List Cameras")

    def _run_capture(self):
        if not self._check_session(): return
        self._run_command([
            PYTHON, "-m", "capture.capture",
            "--config",   self.v_cams.get(),
            "--output",   self.v_session.get(),
            "--n-frames", self.cap_nframes.get(),
        ], "Capture Session")

    def _run_extrinsic(self):
        if not self._check_session(): return
        self._run_command([
            PYTHON, "-m", "extrinsic_solver.solve",
            "--session",         self.v_session.get(),
            "--box-config",      self.v_box.get(),
            "--cameras-config",  self.v_cams.get(),
            "--calibration-dir", self.v_calib.get(),
            "--min-markers",     self.ext_min_markers.get(),
            "--max-reproj-px",   self.ext_max_reproj.get(),
        ], "Extrinsic Solver")

    def _run_detect(self):
        if not self._check_session(): return
        cmd = [
            PYTHON, "-m", "ball_detector.detect",
            "--session",         self.v_session.get(),
            "--cameras-config",  self.v_cams.get(),
            "--calibration-dir", self.v_calib.get(),
            "--min-area",        self.det_min_area.get(),
            "--max-area",        self.det_max_area.get(),
            "--max-fit-residual",self.det_residual.get(),
            "--roi-radius",      self.det_roi_r.get(),
        ]
        if self.det_interactive.get():
            cmd.append("--interactive")
        self._run_command(cmd, "Ball Detector")

    def _run_triangulate(self):
        if not self._check_session(): return
        self._run_command([
            PYTHON, "-m", "triangulation.triangulate",
            "--session",         self.v_session.get(),
            "--cameras-config",  self.v_cams.get(),
            "--calibration-dir", self.v_calib.get(),
        ], "Triangulation")

    def _run_error_prop(self):
        if not self._check_session(): return
        self._run_command([
            PYTHON, "-m", "error_propagation.propagate",
            "--session",         self.v_session.get(),
            "--box-config",      self.v_box.get(),
            "--cameras-config",  self.v_cams.get(),
            "--calibration-dir", self.v_calib.get(),
            "--n-mc",            self.err_nmc.get(),
        ], "Error Propagation")

    def _run_compare(self):
        if not self._check_session(): return
        if not self.v_sim.get().strip():
            messagebox.showerror("Missing path", "Set Sim output path first.")
            return
        self._run_command([
            PYTHON, "-m", "comparison.compare",
            "--session",      self.v_session.get(),
            "--sim-output",   self.v_sim.get(),
            "--box-config",   self.v_box.get(),
            "--threshold-mm", self.cmp_thresh.get(),
            "--sim-rank",     self.cmp_rank.get(),
        ], "Comparison")

    def _run_full_pipeline(self):
        if not self._check_session(): return
        if not self.v_sim.get().strip():
            messagebox.showerror("Missing path", "Set Sim output path first.")
            return
        cmd = [
            PYTHON, "run_pipeline.py",
            "--session",         self.v_session.get(),
            "--sim-output",      self.v_sim.get(),
            "--box-config",      self.v_box.get(),
            "--cameras-config",  self.v_cams.get(),
            "--calibration-dir", self.v_calib.get(),
            "--threshold-mm",    self.fp_thresh.get(),
            "--sim-rank",        self.fp_rank.get(),
            "--min-markers",     self.fp_min_markers.get(),
            "--max-reproj-px",   self.fp_max_reproj.get(),
            "--min-ball-area",   self.fp_min_area.get(),
            "--max-ball-area",   self.fp_max_area.get(),
            "--n-mc",            self.fp_nmc.get(),
        ]
        if self.fp_skip_err.get():
            cmd.append("--skip-error-propagation")
        if self.fp_interactive.get():
            cmd.append("--interactive")
        self._run_command(cmd, "Full Pipeline")


if __name__ == "__main__":
    app = App()
    app.mainloop()
