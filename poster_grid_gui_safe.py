#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import os, csv

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect

def find_poster_quad(gray):
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 40, 140)
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.02*w*h:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4,2).astype("float32"))
    return None

def four_point_transform(image, quad, dst_size=(1000,700)):
    quad = order_points(quad)
    dst_w, dst_h = dst_size
    dst = np.array([[0,0],[dst_w-1,0],[dst_w-1,dst_h-1],[0,dst_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h))
    return warped

class App:
    def __init__(self, root):
        self.root = root
        root.title("Poster Grid Detector (Safe Loader)")

        self.img = None
        self.photo = None

        self.grid_rows = tk.IntVar(value=5)
        self.grid_cols = tk.IntVar(value=5)
        self.use_auto_thresh = tk.BooleanVar(value=True)
        self.thresh = tk.IntVar(value=120)

        top = ttk.Frame(root, padding=8)
        top.grid(row=0, column=0, sticky="nsew")

        ttk.Label(top, text="Image path:").grid(row=0, column=0, sticky="w")
        self.path_var = tk.StringVar(value="")
        self.path_entry = ttk.Entry(top, textvariable=self.path_var, width=80)
        self.path_entry.grid(row=0, column=1, columnspan=4, sticky="we", padx=6)
        ttk.Button(top, text="Load from path", command=self.load_from_path).grid(row=0, column=5, sticky="e")

        ttk.Label(top, text="Rows:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.grid_rows, width=4).grid(row=1, column=1, sticky="w")
        ttk.Label(top, text="Cols:").grid(row=1, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.grid_cols, width=4).grid(row=1, column=3, sticky="w")

        ttk.Checkbutton(top, text="Auto threshold", variable=self.use_auto_thresh, command=self.on_toggle).grid(row=2, column=0, sticky="w")
        self.slider = ttk.Scale(top, from_=0, to=255, orient="horizontal", variable=self.thresh)
        self.slider.grid(row=2, column=1, columnspan=3, sticky="we")

        ttk.Button(top, text="Detect Grid", command=self.detect).grid(row=3, column=0, sticky="w", pady=6)

        self.canvas_w, self.canvas_h = 900, 650
        self.canvas = tk.Canvas(top, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.canvas.grid(row=4, column=0, columnspan=6, pady=8)

        self.out = tk.Text(top, height=8, width=120)
        self.out.grid(row=5, column=0, columnspan=6)

        top.columnconfigure(1, weight=1)
        self.on_toggle()

    def on_toggle(self):
        if self.use_auto_thresh.get():
            self.slider.state(["disabled"])
        else:
            self.slider.state(["!disabled"])

    def load_from_path(self):
        path = self.path_var.get().strip().strip('"').strip("'")
        if not path:
            messagebox.showinfo("Info", "Paste an image path first.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Error", f"File not found:\n{path}")
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read image (unsupported format or path).")
            return
        self.img = img
        self.show_image(img)
        self.out.delete("1.0", tk.END)
        self.out.insert(tk.END, f"Loaded: {path}\n")

    def show_image(self, bgr):
        h, w = bgr.shape[:2]
        scale = min(self.canvas_w / w, self.canvas_h / h, 1.0)
        disp = cv2.resize(bgr, (int(w*scale), int(h*scale)))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def detect(self):
        if self.img is None:
            messagebox.showinfo("Info", "Load an image first.")
            return

        rows = int(self.grid_rows.get())
        cols = int(self.grid_cols.get())

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        quad = find_poster_quad(gray)
        used_fallback = False
        if quad is None:
            used_fallback = True
            h, w = self.img.shape[:2]
            quad = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")

        # warp size based on aspect ratio
        w_src = max(np.linalg.norm(quad[1]-quad[0]), np.linalg.norm(quad[2]-quad[3]))
        h_src = max(np.linalg.norm(quad[2]-quad[1]), np.linalg.norm(quad[3]-quad[0]))
        dst_w = 1000
        dst_h = max(600, int(round(dst_w * (h_src / max(w_src,1)))))
        warped = four_point_transform(self.img, quad, dst_size=(dst_w, dst_h))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        H, W = warped_gray.shape[:2]
        cell_h = H / rows
        cell_w = W / cols

        means = np.zeros((rows, cols), float)
        for r in range(rows):
            for c in range(cols):
                x1, x2 = int(round(c*cell_w)), int(round((c+1)*cell_w))
                y1, y2 = int(round(r*cell_h)), int(round((r+1)*cell_h))
                means[r,c] = float(np.mean(warped_gray[y1:y2, x1:x2]))

        if self.use_auto_thresh.get():
            otsu, _ = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_val = float(otsu) if np.isscalar(otsu) else float(np.mean(warped_gray))
            thr = int(round((float(np.min(means)) + min(otsu_val, float(np.mean(warped_gray)))) / 2))
        else:
            thr = int(self.thresh.get())

        black = [(r,c) for r in range(rows) for c in range(cols) if means[r,c] < thr]

        # draw overlay
        vis = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)
        for r in range(rows):
            for c in range(cols):
                x1, x2 = int(round(c*cell_w)), int(round((c+1)*cell_w))
                y1, y2 = int(round(r*cell_h)), int(round((r+1)*cell_h))
                cv2.rectangle(vis, (x1,y1), (x2,y2), (180,180,180), 1)

        for (r,c) in black:
            x1, x2 = int(round(c*cell_w)), int(round((c+1)*cell_w))
            y1, y2 = int(round(r*cell_h)), int(round((r+1)*cell_h))
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 3)
            cv2.putText(vis, f"{r},{c}", (x1+6, y1+22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        self.show_image(vis)

        self.out.delete("1.0", tk.END)
        self.out.insert(tk.END, f"Poster quad fallback used: {used_fallback}\n")
        self.out.insert(tk.END, f"Threshold: {thr}\n")
        self.out.insert(tk.END, f"Black cells (0-based row,col): {black}\n")
        self.out.insert(tk.END, f"Black cells (1-based row,col): {[(r+1,c+1) for (r,c) in black]}\n\n")
        self.out.insert(tk.END, "Cell mean intensities:\n")
        for r in range(rows):
            self.out.insert(tk.END, "  " + ", ".join(f"{means[r,c]:.1f}" for c in range(cols)) + "\n")

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()