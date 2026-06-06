# from ultralytics import YOLO
# import torch

# print("CUDA:", torch.cuda.is_available())

# model = YOLO(
#     "/userhome/phd/gayathri.rangu/AerialGuardianMOT/scripts/runs/runs/visdrone/E03_person_ft-3/weights/best.pt"
# )
# # model = YOLO("yolo26n.pt")
# # model = YOLO("/userhome/phd/gayathri.rangu/AerialGuardianMOT/scripts/runs/runs/visdrone/E03_person_ft-3/weights/best.pt")
# metrics = model.val(
#     data="visdrone_personval.yaml",
#     imgsz=1024,
#     conf=0.01,
#     iou=0.6
# )

# print(metrics.box.map)
# print(metrics.box.map50)
# print(metrics.box.map75)
# print(metrics.box.mp)
# print(metrics.box.mr)
from ultralytics import YOLO
import torch

print("CUDA:", torch.cuda.is_available())

model = YOLO(
"/userhome/phd/gayathri.rangu/AerialGuardianMOT/scripts/runs/runs/visdrone/E03_person_ft-3/weights/best.pt"
)
SEQ_DIR="/userhome/phd/gayathri.rangu/AerialGuardianMOT/data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v"
results = model.predict(
    source=SEQ_DIR,
    imgsz=1280,
    conf=0.25,
    device=0,

    save=True,          # ← IMPORTANT
    save_txt=True,
    save_conf=True,

    show=False
)
print("DONE")