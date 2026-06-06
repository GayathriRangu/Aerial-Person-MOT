# import os
# from collections import Counter

# ann_dir = "../data/VisDrone2019-MOT-train/annotations"

# counter = Counter()

# for file in os.listdir(ann_dir):

#     with open(os.path.join(ann_dir,file)) as f:

#         for line in f:

#             cls = int(line.strip().split(",")[7])

#             counter[cls] += 1

# print(counter)
################count num of files generated total#############
# import os

# count = 0

# for root, dirs, files in os.walk(
#     "../data/VisDrone2019-MOT-train/yolo_labels"
# ):
#     count += len(files)

# print(count)

#######create image lists for train and val################
import os

TRAIN_ROOT = "../data/VisDrone2019-MOT-train/sequences"
VAL_ROOT   = "../data/VisDrone2019-MOT-val/sequences"

def make_list(root_dir, outfile):

    image_paths = []

    for seq in sorted(os.listdir(root_dir)):

        seq_dir = os.path.join(root_dir, seq)

        if not os.path.isdir(seq_dir):
            continue

        for img in sorted(os.listdir(seq_dir)):

            if img.endswith(".jpg"):

                image_paths.append(
                    os.path.abspath(
                        os.path.join(seq_dir, img)
                    )
                )

    with open(outfile, "w") as f:
        f.write("\n".join(image_paths))

    print(
        outfile,
        len(image_paths)
    )

make_list(
    TRAIN_ROOT,
    "../data/train.txt"
)

make_list(
    VAL_ROOT,
    "../data/val.txt"
)