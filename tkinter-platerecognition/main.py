import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import easyocr
from tracker import *

model = YOLO("./utils/last.pt")
names = ["Mobil Bus", "Mobil MPV", "Mobil Pickup", "Plat-Nomor", "Mobil Polisi", "Mobil Sedan", "Mobil Truk",""]
tracker = Tracker()

root = ttk.Window(themename="solar")
window = ttk.Frame(root, padding=10)
window.pack(fill=BOTH, expand=YES)
style = ttk.Style()
theme_names = style.theme_names()

frame_header = ttk.Frame(window, padding=(10, 10, 10, 0))
frame_header.pack(fill=X, expand=YES)
theme_selected = ttk.Label(
    master=frame_header,
    text="ANPR-Automatic number-plate recognition",
    font="-size 12 -weight bold",
)
theme_selected.pack(side=LEFT)

lbl = ttk.Label(
    frame_header, text="Sugeng Wahyu Widodo", font="arial 10 bold", bootstyle="danger"
)
imgUsrCap = Image.open("assets/image/user.png")
imgUsrRez = imgUsrCap.resize((50, 50), Image.LANCZOS)
imgUsr = ImageTk.PhotoImage(imgUsrRez)
ttk.Label(frame_header, image=imgUsr).pack(side=RIGHT)
lbl.pack(side=RIGHT)


ttk.Separator(window).pack(fill=X, pady=10, padx=10)

frame_left = ttk.Frame(window, padding=5)
frame_left.pack(side=LEFT, fill=BOTH, expand=YES)
frame_right = ttk.Frame(window, padding=5)
frame_right.pack(side=RIGHT, fill=BOTH, expand=YES)
frame_center = ttk.Frame(window, padding=5)
frame_center.pack(side=RIGHT, fill=BOTH, expand=YES)

color_group = ttk.Labelframe(master=frame_left, text="Camera Live", padding=10)
color_group.pack(fill=X, side=TOP)

control_list = ttk.Labelframe(
    master=frame_center, text="Car Registration", padding=(10, 5)
)
control_list.pack(fill=X)

table_data = [
    (1, "South Island, New Zealand", "BE980980"),
    (2, "Paris", "BE980980"),
    (3, "Bora Bora", "BE980980"),
    (4, "Maui", "BE980980"),
    (5, "Tahiti", "BE9809XX"),
]

tv = ttk.Treeview(master=control_list, columns=[0, 1, 2], show=HEADINGS, height=22)
for row in table_data:
    tv.insert("", END, values=row)

tv.selection_set("I001")
tv.heading(0, text="No.")
tv.heading(1, text="Jenis Kendaraan")
tv.heading(2, text="Nomor Plat")
tv.column(0, width=40, anchor=CENTER)
tv.column(1, width=200)
tv.column(2, width=100, anchor=CENTER)
tv.pack(side=LEFT, anchor=NE, fill=X)

radio1 = ttk.Radiobutton(color_group, text="RGB", value=1)
radio1.grid(column=0, row=0, padx=10, pady=10)
radio1.invoke()
radio2 = ttk.Radiobutton(color_group, text="GRAY", value=2)
radio2.grid(column=1, row=0, padx=10, pady=10)
radio2.invoke()
radio3 = ttk.Radiobutton(color_group, text="HSV", value=3)
radio3.grid(column=2, row=0, padx=10, pady=10)
radio3.invoke()

imagenBI = Image.open("assets/image/bg.jpg")
picResize = imagenBI.resize((470, 312), Image.LANCZOS)
xdx = ImageTk.PhotoImage(picResize)
grb = ttk.Label(color_group, image=xdx)
grb.grid(column=0, row=1, columnspan=3)

control_group = ttk.Labelframe(master=frame_right, text="Data Reader", padding=(10, 5))
control_group.pack(fill=X)

ttk.Label(master=control_group, text="Jumlah Kendaraan").grid(column=0, row=0, pady=10)
inp_count = ttk.Label(
    master=control_group,
    font=("arial 10 bold"),
    width=25,
    text="20",
    bootstyle="inverse-info",
)
inp_count.grid(column=1, row=0, padx=10, sticky="w")

ttk.Label(master=control_group, text="Type Kendaraan").grid(
    column=0, row=1, pady=10, sticky="w"
)
inp_type = ttk.Label(
    master=control_group,
    font=("arial 10 bold"),
    width=25,
    text="Mobil Truck",
    bootstyle="inverse-info",
)
inp_type.grid(column=1, row=1, padx=10, sticky="w")

ttk.Label(master=control_group, text="Nomor Plat").grid(
    column=0, row=2, pady=10, sticky="w"
)
inp_plat = ttk.Label(
    master=control_group,
    font=("arial 10 bold"),
    width=25,
    text="BE***",
    bootstyle="inverse-info",
)
inp_plat.grid(column=1, row=2, padx=10, pady=10, sticky="w")


imgPlateCp = Image.open("assets/image/bg.jpg")
imgPlateRez = imgPlateCp.resize((175, 50), Image.LANCZOS)
imgPlate = ImageTk.PhotoImage(imgPlateRez)
inp_imgPlat = ttk.Label(master=control_group, image=imgPlate, bootstyle="inverse-info")
inp_imgPlat.grid(column=1, row=3, padx=10, sticky="w")

control_btn = ttk.Labelframe(master=frame_right, text="Data Reader", padding=(10, 5))
control_btn.pack(fill=X, side=BOTTOM)

btn_play = ttk.Button(master=control_btn, text="ANPR Start", bootstyle="btn-info")
btn_play.grid(column=0, row=0, padx=5, pady=5, sticky="w")

btn_close = ttk.Button(master=control_btn, text="Close Rec")
btn_close.grid(column=1, row=0, padx=5, pady=5, sticky="s")

window.mainloop()