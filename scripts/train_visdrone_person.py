from ultralytics import YOLO
import torch

print("="*60)
print("VISDRONE PERSON FINETUNING")
print("="*60)

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

MODEL = "yolo26n.pt"

model = YOLO(MODEL)

results = model.train(

    # Dataset
    data="../data/visdrone_person.yaml",

    # Training
    epochs=50,
    imgsz=1024,

    # Hardware
    batch=16,
    workers=4,
    device=0,

    # Optimization
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,

    # Transfer learning
    pretrained=True,

    # Memory
    cache=False,

    # Augmentations
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    degrees=0.0,
    translate=0.1,
    scale=0.5,

    fliplr=0.5,

    mosaic=1.0,
    mixup=0.1,

    # Save
    project="../runs/visdrone",
    name="E03_person_ft",

    save=True,
    save_period=10,

    plots=True,
    val=True,

    verbose=True
)

print("\nTRAINING COMPLETE")
print(results)
