"""
gui.py — Acoustic Levitation Measurement System GUI

Launch:
    python gui.py
"""

from __future__ import annotations

import os
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
        self._active_jobs = 0
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
        self._tab_box_cal()
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

        # Camera selector — populated from cameras.yaml
        ttk.Label(f, text="Camera:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        self._cal_cam_var = tk.StringVar()
        self._cal_cam_combo = ttk.Combobox(f, textvariable=self._cal_cam_var,
                                           state="readonly", width=26)
        self._cal_cam_combo.grid(row=0, column=1, sticky="w", pady=3)
        self._cal_cam_combo.bind("<<ComboboxSelected>>", self._on_cal_cam_selected)
        ttk.Button(f, text="↺", width=3,
                   command=self._reload_cal_cameras).grid(row=0, column=2, padx=(4, 0))

        self.cal_imgs   = self._browse(f, 1, "Images dir",       kind="dir")
        self.cal_id     = self._field(f,  2, "Camera ID",         "")
        self.cal_out    = self._field(f,  3, "Output YAML",       "")
        self.cal_sq_x   = self._field(f,  4, "Squares X",         "8")
        self.cal_sq_y   = self._field(f,  5, "Squares Y",         "11")
        self.cal_sq_len = self._field(f,  6, "Square length (m)", "0.015")
        self.cal_mk_len = self._field(f,  7, "Marker length (m)", "0.011")
        self.cal_dict   = self._field(f,  8, "ArUco dict",        "DICT_4X4_50")
        self.cal_reproj = self._field(f,  9, "Max reproj (px)",   "1.0")

        ttk.Button(f, text="▶  Run Calibration",
                   command=self._run_calibrate).grid(row=10, column=0, columnspan=3,
                                                     pady=14, ipadx=10, ipady=4)

    def _reload_cal_cameras(self):
        """Read cameras.yaml and refresh the camera dropdown."""
        path = self.v_cams.get().strip()
        if not path:
            messagebox.showerror("Missing path", "Set Cameras config in Common Paths first.")
            return
        try:
            import yaml
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
            ids = [c["id"] for c in cfg.get("cameras", [])]
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return
        self._cal_cam_combo["values"] = ids
        self._cal_cam_cfg = cfg  # cache for _on_cal_cam_selected
        if ids:
            self._cal_cam_combo.current(0)
            self._on_cal_cam_selected()

    def _on_cal_cam_selected(self, _event=None):
        """Auto-fill Camera ID and Output YAML from the selected camera entry."""
        cfg = getattr(self, "_cal_cam_cfg", None)
        if cfg is None:
            return
        selected = self._cal_cam_var.get()
        for cam in cfg.get("cameras", []):
            if cam["id"] == selected:
                self.cal_id.set(cam["id"])
                self.cal_out.set(cam.get("intrinsics_file", ""))
                break

    # ── Tab 1b: Box marker calibration ───────────────────────────────────────

    def _tab_box_cal(self):
        f = ttk.Frame(self.nb, padding=12)
        self.nb.add(f, text="1b · Box Cal")

        # Camera selector
        ttk.Label(f, text="Camera:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        self._boxcal_cam_var = tk.StringVar()
        self._boxcal_cam_combo = ttk.Combobox(f, textvariable=self._boxcal_cam_var,
                                              state="readonly", width=26)
        self._boxcal_cam_combo.grid(row=0, column=1, sticky="w", pady=3)
        self._boxcal_cam_combo.bind("<<ComboboxSelected>>", self._on_boxcal_cam_selected)
        ttk.Button(f, text="↺", width=3,
                   command=self._reload_boxcal_cameras).grid(row=0, column=2, padx=(4, 0))

        self.boxcal_imgs   = self._browse(f, 1, "Images dir",          kind="dir")
        self.boxcal_intr   = self._field(f,  2, "Intrinsics YAML",     "")
        self.boxcal_out    = self._field(f,  3, "Output config",        "config/box.yaml")
        self.boxcal_min_mk = self._field(f,  4, "Min markers",          "3")
        self.boxcal_reproj = self._field(f,  5, "Max reproj (px)",      "1.5")
        self.boxcal_dbg    = self._browse(f, 6, "Debug dir (optional)", kind="dir")

        ttk.Label(f, text="Box config taken from Common Paths above.",
                  foreground="gray").grid(row=7, column=0, columnspan=3,
                                          sticky="w", pady=(8, 0))

        ttk.Button(f, text="▶  Run Box Calibration",
                   command=self._run_box_cal).grid(row=8, column=0, columnspan=3,
                                                   pady=14, ipadx=10, ipady=4)

    def _reload_boxcal_cameras(self):
        path = self.v_cams.get().strip()
        if not path:
            messagebox.showerror("Missing path", "Set Cameras config in Common Paths first.")
            return
        try:
            import yaml
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
            ids = [c["id"] for c in cfg.get("cameras", [])]
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return
        self._boxcal_cam_combo["values"] = ids
        self._boxcal_cam_cfg = cfg
        if ids:
            self._boxcal_cam_combo.current(0)
            self._on_boxcal_cam_selected()

    def _on_boxcal_cam_selected(self, _event=None):
        cfg = getattr(self, "_boxcal_cam_cfg", None)
        if cfg is None:
            return
        selected = self._boxcal_cam_var.get()
        for cam in cfg.get("cameras", []):
            if cam["id"] == selected:
                self.boxcal_intr.set(cam.get("intrinsics_file", ""))
                break

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

        self.det_bg_mode = self._bg_section(f, 6)

        ttk.Button(f, text="▶  Run Ball Detector",
                   command=self._run_detect).grid(row=7, column=0, columnspan=3,
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

        self.fp_bg_mode = self._bg_section(f, 9)

        ttk.Button(f, text="▶▶  Run Full Pipeline",
                   command=self._run_full_pipeline).grid(
            row=10, column=0, columnspan=3, pady=18, ipadx=24, ipady=8)

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

    def _bg_section(self, parent, row: int) -> tk.StringVar:
        """Background subtraction radio group (per-camera — no single path)."""
        lf = ttk.LabelFrame(parent, text="Background Subtraction", padding=6)
        lf.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        mode_var = tk.StringVar(value="none")

        hint_label = ttk.Label(lf, foreground="gray", justify="left")

        def _toggle(*_):
            if mode_var.get() == "file":
                hint_label.config(
                    text="Place background.png in each camera's frame folder:\n"
                         "  <session>/<cam_id>/background.png\n"
                         "Each camera needs its own reference shot (levitator on, no ball)."
                )
                hint_label.pack(anchor="w", pady=(4, 0))
            else:
                hint_label.pack_forget()

        for val, text in [
            ("none",   "None  (threshold raw frame)"),
            ("file",   "Per-camera background images  (one reference shot per camera)"),
            ("median", "Median of frames  (computed from captured frames automatically)"),
        ]:
            ttk.Radiobutton(lf, text=text, variable=mode_var,
                            value=val, command=_toggle).pack(anchor="w")

        return mode_var

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

        bottom = ttk.Frame(lf)
        bottom.pack(fill="x", pady=(4, 0))

        self._status_text = tk.StringVar(value="Idle")
        ttk.Label(bottom, textvariable=self._status_text,
                  anchor="w", width=36).pack(side="left", padx=(0, 8))
        self._progress = ttk.Progressbar(bottom, mode="indeterminate", length=180)
        self._progress.pack(side="left")
        ttk.Button(bottom, text="Clear log",
                   command=self._clear_log).pack(side="right")

    def _log(self, text: str, tag: str = "") -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text, tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ── Progress / status ─────────────────────────────────────────────────────

    def _job_start(self, header: str) -> None:
        self._active_jobs += 1
        self._status_text.set(f"Running: {header} …")
        if self._active_jobs == 1:
            self._progress.start(12)

    def _job_done(self) -> None:
        self._active_jobs = max(0, self._active_jobs - 1)
        if self._active_jobs == 0:
            self._progress.stop()
            self._progress["value"] = 0
            self._status_text.set("Idle")
        else:
            self._status_text.set(f"{self._active_jobs} job(s) running …")

    # ── Command runner (non-blocking) ─────────────────────────────────────────

    def _check_session(self) -> bool:
        if not self.v_session.get().strip():
            messagebox.showerror("Missing path", "Set Session dir first.")
            return False
        return True

    def _run_command(self, cmd: list[str], header: str) -> None:
        self._log(f"\n{'─' * 58}\n  {header}\n{'─' * 58}\n", "hdr")
        self._job_start(header)

        def _worker():
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(ROOT),
                    env=env,
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
            finally:
                self.after(0, self._job_done)

        threading.Thread(target=_worker, daemon=True).start()

    # ── Step runners ─────────────────────────────────────────────────────────

    def _run_calibrate(self):
        cmd = [
            PYTHON, "-m", "intrinsic_calibration.calibrate",
            "--camera-id",       self.cal_id.get(),
            "--images-dir",      self.cal_imgs.get(),
            "--squares-x",       self.cal_sq_x.get(),
            "--squares-y",       self.cal_sq_y.get(),
            "--square-length",   self.cal_sq_len.get(),
            "--marker-length",   self.cal_mk_len.get(),
            "--dict",            self.cal_dict.get(),
            "--max-reproj-px",   self.cal_reproj.get(),
        ]
        # prefer explicit output field; fall back to deriving from cameras.yaml
        if self.cal_out.get().strip():
            cmd += ["--output", self.cal_out.get()]
        elif self.v_cams.get().strip():
            cmd += ["--cameras-config", self.v_cams.get()]
        else:
            messagebox.showerror("Missing path",
                                 "Set Output YAML or load a camera from Cameras config.")
            return
        self._run_command(cmd, "Intrinsic Calibration")

    def _run_box_cal(self):
        cmd = [
            PYTHON, "-m", "box_calibration.calibrate",
            "--images-dir",    self.boxcal_imgs.get(),
            "--intrinsics",    self.boxcal_intr.get(),
            "--box-config",    self.v_box.get(),
            "--output",        self.boxcal_out.get(),
            "--min-markers",   self.boxcal_min_mk.get(),
            "--max-reproj-px", self.boxcal_reproj.get(),
        ]
        if self.boxcal_dbg.get().strip():
            cmd += ["--debug-dir", self.boxcal_dbg.get()]
        self._run_command(cmd, "Box Marker Calibration")

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
        if self.det_bg_mode.get() == "median":
            cmd.append("--median-background")
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
        if self.fp_bg_mode.get() == "median":
            cmd.append("--median-background")
        self._run_command(cmd, "Full Pipeline")


if __name__ == "__main__":
    app = App()
    app.mainloop()
