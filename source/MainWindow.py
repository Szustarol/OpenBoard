import tkinter as tk

import numpy as np
import tensorflow.keras as keras
from PIL import Image, ImageTk

from board_classifier import predict_classes
from perspective_correction import correct_perspective
from webcam_pooler import WebcamPooler

segmentation_model = keras.models.load_model("segmentation_model.h5")

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OpenBoard")

        self.data_frame = tk.Frame(self)
        self.data_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.capture_frame = tk.Frame(self.data_frame)
        self.capture_frame.grid(column=0, row=0, columnspan=3, rowspan=2)

        self.capture_canvas = self._build_canvas_with_label(self.capture_frame, "Camera input",
        canvas_height=420, canvas_width=700)

        self.process_and_input_frame = tk.Frame(self.data_frame)
        self.process_and_input_frame.grid(column=0, row=2, columnspan=3)

        self.capture_raw_canvas = self._build_canvas_with_label(self.process_and_input_frame, "Model input", side=tk.LEFT)
        self.segmentation_canvas = self._build_canvas_with_label(self.process_and_input_frame, "Segmentation", side=tk.LEFT)
        self.classification_canvas = self._build_canvas_with_label(self.process_and_input_frame, "Classifier input", side=tk.LEFT)

        self.input_output_frame = tk.Frame(self)
        self.input_output_frame.pack(side=tk.LEFT)

        self.capture_button = tk.Button(self.input_output_frame, text="Capture", command=self.run_models)
        self.capture_button.pack(side=tk.TOP)

        self.fen_output = tk.Text(self.input_output_frame, height=1, borderwidth=0, width=70)
        self.fen_output.insert(1.0, "FEN will be displayed here")
        self.fen_output.pack(side=tk.TOP)
        self.fen_output.configure(state="disabled")
        self.fen_output.configure(inactiveselectbackground=self.fen_output.cget("selectbackground"))
        
        self.canvas_output = self._build_canvas_with_label(self.input_output_frame, "Model output", side=tk.TOP)

    def run_models(self):
        self.model_input_image = self.image.copy()
        self.frame_image = self.model_input_image.resize((256,256))
        self.frame_image_tk = ImageTk.PhotoImage(image = self.frame_image)
        self.capture_raw_canvas.delete("all")
        self.capture_raw_canvas.create_image(0, 0, image=self.frame_image_tk, anchor=tk.NW)

        model_input_image = self.model_input_image.resize((512, 512))
        model_input_image = np.array(model_input_image)/255
        model_input_image = np.expand_dims(model_input_image, axis=0)
        segmentation_output = segmentation_model(model_input_image)
        segmentation_image = Image.fromarray(np.squeeze(segmentation_output)*255).resize((256, 256))
        self.segmentation_image_tk = ImageTk.PhotoImage(image=segmentation_image) 
        self.segmentation_canvas.delete("all")
        self.segmentation_canvas.create_image(0, 0, image=self.segmentation_image_tk, anchor=tk.NW)

        mask_bin = np.where(np.squeeze(segmentation_output) > 0.5, 255, 0).astype(np.uint8)

        _, corrected_image = correct_perspective(np.squeeze(model_input_image), mask_bin)
        corrected_image = Image.fromarray(np.squeeze(corrected_image*255).astype(np.uint8))
        
        self.corrected_image_tk = ImageTk.PhotoImage(image=corrected_image.resize((256, 256)))
        self.classification_canvas.delete("all")
        self.classification_canvas.create_image(0, 0, image=self.corrected_image_tk, anchor=tk.NW)

        fen, board_image = predict_classes(corrected_image)

        self.board_output_image = ImageTk.PhotoImage(board_image.resize((256, 256)))
        self.canvas_output.delete("all")
        self.canvas_output.create_image(0, 0, image=self.board_output_image, anchor=tk.NW)

        self.fen_output.configure(state="normal")
        self.fen_output.delete(1.0, tk.END)
        self.fen_output.insert(1.0, f"{fen}")
        self.fen_output.configure(state="disabled")

    def set_camera_image(self, image):
        self.image = image.copy()
        image = image.resize((self.capture_canvas.winfo_width(), self.capture_canvas.winfo_height()))
        self.img_tk = ImageTk.PhotoImage(image = image)
        self.capture_canvas.delete("all")
        self.capture_canvas.create_image(0, 0, image=self.img_tk, anchor=tk.NW)

    def _build_canvas_with_label(self, frame, text, canvas_width=256, canvas_height=256, **pack_args):
        child_frame = tk.Frame(frame)
        child_frame.pack(**pack_args)
        tk.Label(child_frame, text=text).pack()
        canvas = tk.Canvas(child_frame, width=canvas_width, height=canvas_height)
        canvas.pack()
        return canvas
            
if __name__ == "__main__":
    app = MainWindow()
    pooler = WebcamPooler(app.set_camera_image)
    pooler.start()

    def on_exit():
        pooler.stop()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_exit)
    app.mainloop()