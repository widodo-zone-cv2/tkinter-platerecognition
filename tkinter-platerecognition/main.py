import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
from PIL import Image, ImageTk
import easyocr
from tracker import *
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

model = YOLO("./utils/last.pt")
names = ["Mobil Bus", "Mobil MPV", "Mobil Pickup", "Plat-Nomor", "Mobil Polisi", "Mobil Sedan", "Mobil Truk",""]
tracker = Tracker()

def setup(master):
    global color_group,gbr
    global inp_count,inp_type,tv,inp_plat,inp_imgPlat
    root = ttk.Frame(master, padding=10)
    root.pack(fill=BOTH, expand=YES)
    style = ttk.Style()
    # theme_names = style.theme_names()
    frame_header = ttk.Frame(root, padding=(10, 10, 10, 0))
    frame_header.pack(fill=X, expand=YES)
    ttk.Label(
        master=frame_header,
        text="ANPR-Automatic number-plate recognition",
        font="-size 12 -weight bold",
    ).pack(side=LEFT)
    lbl = ttk.Label(
        frame_header,
        text="Sugeng Wahyu Widodo",
        font="arial 10 bold",
        bootstyle="danger"
    )


    ttk.Label(frame_header, image=image_user).pack(side=RIGHT)
    lbl.pack(side=RIGHT)

    # Separator antar header dan frame Main menu
    ttk.Separator(root).pack(fill=X, pady=10, padx=10)
    frame_left = ttk.Frame(root, padding=5)
    frame_left.pack(side=LEFT, fill=BOTH, expand=YES)
    frame_right = ttk.Frame(root, padding=5)
    frame_right.pack(side=RIGHT, fill=BOTH, expand=YES)
    frame_center = ttk.Frame(root, padding=5)
    frame_center.pack(side=RIGHT, fill=BOTH, expand=YES)

    color_group = ttk.Labelframe(master=frame_left, text="Camera Live", padding=10)
    color_group.pack(fill=X, side=TOP)

    control_list = ttk.Labelframe(
        master=frame_center, text="Car Registration", padding=(10, 5)
    )
    control_list.pack(fill=X)

    tv = ttk.Treeview(master=control_list, columns=[0, 1, 2], show=HEADINGS, height=22)
    tv.heading(0, text="No.")
    tv.heading(1, text="Jenis Kendaraan")
    tv.heading(2, text="Nopol")
    tv.column(0, width=30, anchor=CENTER)
    tv.column(1, width=100)
    tv.column(2, width=70, anchor=CENTER)
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


    gbr = ttk.Label(color_group)
    gbr.grid(column=0, row=1, columnspan=3)

    control_group = ttk.Labelframe(master=frame_right, text="Data Reader", padding=(10, 5))
    control_group.pack(fill=X)

    ttk.Label(master=control_group, text="Jumlah Kendaraan").grid(column=0, row=0, pady=10)
    inp_count = ttk.Label(
        master=control_group,
        font=("arial 10 bold"),
        width=25,
        text="0",
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
        bootstyle="inverse-info",
    )
    inp_plat.grid(column=1, row=2, padx=10, pady=10, sticky="w")

    # imgPlateCp = Image.open("assets/image/bg.jpg")
    # imgPlateRez = imgPlateCp.resize((175, 50), Image.LANCZOS)
    # imgPlate = ImageTk.PhotoImage(imgPlateRez)
    inp_imgPlat = ttk.Label(master=control_group, bootstyle="inverse-info")
    inp_imgPlat.grid(column=1, row=3, padx=10, sticky="w")

    control_btn = ttk.Labelframe(master=frame_right, text="Data Reader", padding=(10, 5))
    control_btn.pack(fill=X, side=BOTTOM)

    btn_play = ttk.Button(master=control_btn,
                          text="ANPR Start",
                          bootstyle="btn-info",
                          command=main_video)
    btn_play.grid(column=0, row=0, padx=5, pady=5, sticky="w")

    btn_close = ttk.Button(master=control_btn, text="Close Rec")
    btn_close.grid(column=1, row=0, padx=5, pady=5, sticky="s")
    return root

def show_pic_plate(x=0):
    global plscl,plres,image_plate
    if runct:
        plscl = x
    else:
        plscl = Image.open("./assets/image/user.png")
    plres = plscl.resize((175, 50), Image.LANCZOS)
    image_plate = ImageTk.PhotoImage(plres)
    inp_imgPlat.configure(image=image_plate)
def platDetect():
    global reader
    grayPlate = cv2.cvtColor(cap_plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayPlate, 64, 255, cv2.THRESH_BINARY_INV)
    output = reader.readtext(thresh)
    imx = Image.fromarray(cap_plate)
    show_pic_plate(imx)
    wak =''
    for out in output:
        text_bbox, text, text_score = out
        if text_score < 0.4:
            continue
        nomor = ''
        jadi = ''
        for asu in out[1]:
            nomor += asu
        jadi += nomor
        wak += jadi
    inp_plat.configure(text=wak)
    print(f"fer= {wak}")
def recognition():
    global hit,counter,cap_plate
    area = [(134, 181), (100, 200), (284, 269), (302, 236)]
    results = model(video,stream=True)
    wedus = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
    for r in results:
        boxes = r.boxes
        daftar = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cls = int(box.cls[0].item())
            daftar.append([x1, y1, x2, y2, cls])
            if cls != 3:
                cvzone.cornerRect(video, (x1, y1, w, h), colorR=(255,255,0),t=2,l=20)
                cvzone.putTextRect(video, f'{names[cls]}', (max(0, x1), max(35, y1)), scale=0.9, thickness=1, colorR=(0, 0, 0),offset=3)
        bbox_id = tracker.update(daftar)
        for bbox in bbox_id:
            x3, y3, x4, y4, id, kelas = bbox
            w1, h1 = x4 - x3, y4 - y3
            cx = int(x3 + x4) // 2
            cy = int(y4)
            if kelas == 3:
                cv2.rectangle(video, (x3, y3, w1, h1), (0, 0, 255), 1)
            else:
                cv2.circle(video, (cx, cy), 1, (255, 255, 0), -2)
            ress = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
            if ress >= 0:
                if kelas != 3:
                    if id != counter:
                        hit +=1
                        tv.insert("", END, values=(hit,names[kelas],''))
                        tv.selection_set("I001")
                        inp_type.configure(text=names[kelas])
                        counter =id
                    inp_count.configure(text=hit)
                else:
                    cv2.circle(video, (cx, cy), 3, (0, 0, 255), -1)
                    cap_plate = wedus[y3: y3 + h1, x3:x3 + w1]
                    platDetect()
    cv2.polylines(video, [np.array(area, np.int32)], True, (0, 213, 255), 1)

def show_pic_profil():
    global picscl,picres,image_user
    picscl = Image.open("./assets/image/user.png")
    picres = picscl.resize((50, 50), Image.LANCZOS)
    image_user = ImageTk.PhotoImage(picres)

def run_camera():
    global camscl,camrez,camera,video
    global runct,im,gbr
    if runct:
        if cap is not None:
            succes, video = cap.read()
            if succes == True:
                video = cv2.resize(video, (470, 312))
                recognition()
                video = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(video)
                camera = ImageTk.PhotoImage(image=im)
                gbr.configure(image=camera)
                gbr.after(1, run_camera)
            else:
                cap.release()
                cv2.destroyAllWindows()
    else:
        camscl = Image.open("assets/image/bg.jpg")
        camres = camscl.resize((470, 312), Image.LANCZOS)
        camera = ImageTk.PhotoImage(camres)
        gbr.configure(image=camera)
        runct=1

def main_video():
    global cap
    print('player')
    cap = cv2.VideoCapture("assets/videos/cc1.mp4")
    run_camera()
def show_img_plate():
    print('hello')


#Variable GLobal
reader = easyocr.Reader(['en'])
counter = -1
hit =0
runct =0
rec_cars =[]

if __name__ == "__main__":
    app = ttk.Window(
        title="Automatic number-plate recognition",
        themename="solar",
        resizable=(False, False),
    )
    show_pic_profil()
    setup(app)
    show_pic_plate()
    run_camera()
    app.mainloop()
