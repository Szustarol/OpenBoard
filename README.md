# OpenBoard
A free and open source solution for digital chess board scanning.

### How does OpenBoard work?
OpenBoard at it's core is a three-stage machine learning model. The first model detects the chessboard's position on the photo (or screenshot), by creating a segmentation mask. The second stage consists of fitting a quadrilateral to the segmentation, after some preprocessing (keeping only the largest component etc.), and finding a perspective transform that maps the board to a 512x512 square. 
After the mapping is performed, the third stage, and the second model kicks in, which classifies each 64x64 tile of the aforementioned square, outputting either a figure ID or an empty square's ID. All of the classified tiles are then collected and a FEN notation is generated for the supplied photo.

# The models
The pretrained models trained on the ![OpenBoard Dataset](https://github.com/Szustarol/OpenBoard-Dataset), are part of the software and are updated and delivered along with the source.   

Classifier, which takes 64x64 images as input and classifies them into a chess figure, or empty space is available ![here](https://github.com/Szustarol/OpenBoard/blob/main/source/figure_classifier.h5).  
  
The segmentation model, which takes a 512x512 input and outputs a segmentation mask of where it thinks the chessboard is, is available ![here](https://github.com/Szustarol/OpenBoard/blob/main/source/segmentation_model.h5).

# The tools
Multiple scripts assisted making this software, and all of them are bundled with this software, in the `tools/` directory.   
Here is a quick rundown of the available scripts: 
 - `tools/ls_json_to_dict.py` - converts Label Studio label format to a more readable .json representation that is on par with the one in the OpenBoard Dataset repository.
 - `tools/generate_segmentation_dataset.py` - generates segmentation data from the raw labels and the parsed .json label representation.
 - `tools/train_classification_model.ipynb` - code for training the classification model with all the preprocessing.
 - `tools/train_segmentation_model.ipynb` - code for training the segmentation model with all the preprocessing.
 - `tools/perspective_correction.ipynb` - experimentation with the perspective correction algorithm, and code for slicing corrected images into 64x64 tiles that are later used in classification.
