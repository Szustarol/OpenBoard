import json
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def geneate_segmetnation_dataset(labels, source_images_directory, out_dir):
    try:
        shutil.rmtree(out_dir)
    except:
        pass
    os.makedirs(out_dir)
    for image_file, image_label in labels.items():
        source_image = Image.open(os.path.join(source_images_directory,
                                                   image_file))
        source_image = ImageOps.exif_transpose(source_image)

        target_mask = Image.new('RGB', source_image.size, 0)
        board_points = [(int(p['x']*source_image.width), int(p['y']*source_image.height)) for p in image_label['Board']]
        draw = ImageDraw.Draw(target_mask)
        draw.polygon(board_points, fill="#ffffff", outline=None)

        fname, _ = os.path.splitext(image_file)
        outname = fname+'.png'
        outname_label = fname+'.label.png'

        source_image.save(os.path.join(out_dir, outname))
        target_mask.convert("L").save(os.path.join(out_dir, outname_label))




def print_usage():
    print("Usage: python generate_segmentation_Dataset <label json path> <source images path> <output dataset path>")


if __name__ == '__main__':
    try:
        labels_name = sys.argv[1]
        source_images_name = sys.argv[2]
        data_name = sys.argv[3]
    except:
        print_usage()
        exit()

    with open(labels_name, 'r') as labels_file:
        labels = json.load(labels_file)

    geneate_segmetnation_dataset(labels, source_images_name, data_name)
