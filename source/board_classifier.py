import json

import fentoboardimage
import numpy as np
import tensorflow.keras as keras
from PIL import Image

with open("class_map.json", 'r') as cmap_file:
    class_map = json.load(cmap_file)

class_map = {int(idx): val for idx, val in class_map.items()}

figure_classifier = keras.models.load_model("figure_classifier.h5")

pieceset = fentoboardimage.loadPiecesFolder("./gioco")

class_name_to_fen = {
    "White pawn": "P",
    "White bishop": "B",
    "White rook": "R",
    "White king": "K",
    "White queen": "Q",
    "White knight": "N",
    "Black pawn": "p",
    "Black bishop": "b",
    "Black rook": "r",
    "Black king": "k",
    "Black queen": "q",
    "Black knight": "n"
}

def make_fen(board_classes):
    out = ""

    for y in range(8):
        n_spaces = 0
        for x in range(8):
            class_name = class_map[board_classes[y, x]]
            if class_name == 'Empty':
                n_spaces += 1
            else:
                class_symbol = class_name_to_fen[class_name]
                if n_spaces > 0:
                    out = out + str(n_spaces) + class_symbol
                    n_spaces = 0
                else:
                    out = out + class_symbol
        if n_spaces > 0:
            out = out + str(n_spaces)
        if y != 7:
            out = out + "/"

    out = out + " w KQkq - 0 1"

    return out
                



def predict_classes(image):
    image_array = np.asarray(image)

    pieces = np.zeros((8, 8, 64, 64, 3))

    window_w = image.width//8
    window_h = image.height//8

    for y in range(8):
        for x in range(8):
            pieces[y, x, ...] = image_array[y*window_h:(y+1)*window_h,x*window_w:(x+1)*window_w]

    pieces_batch = pieces.reshape((64, 64, 64, 3))
    classifier_output = figure_classifier(pieces_batch).numpy()
    class_output = np.argmax(classifier_output, axis=1)
    class_output = class_output.reshape((8, 8))

    board_fen = make_fen(class_output)

    board_image = fentoboardimage.fenToImage(
        fen=board_fen,
        squarelength=256//8,
        pieceSet=pieceset,
        darkColor="#D18B47",
	    lightColor="#FFCE9E"
    )

    return board_fen, board_image
    