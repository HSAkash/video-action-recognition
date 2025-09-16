# ai.py
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

class Classifier:
    def __init__(self, model_rel_path="model.keras"):
        self.IMAGE_SIZE = 64
        self.SEQUENCE_LENGTH = 20
        self.threshold = .60

        model_path = Path(__file__).resolve().parent / model_rel_path
        self.model = tf.keras.models.load_model(model_path)
        # rolling buffer for 20 frames
        self.seq_imgs = np.zeros(
            (1, self.SEQUENCE_LENGTH, self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
            dtype=np.float32
        )

        self.class_names = [
            'Arguing-Fighting', 'Carrying a laptop', 'Discussing with colleagues', 'Drinking',
            'Eating', 'Giving a presentation', 'Hugging', 'Kissing', 'Looking out the window',
            'Picking', 'Reading', 'Running', 'Shaking hands', 'Sitting down', 'Sleeping',
            'Standing up', 'Throwing', 'Typing on a keyboard', 'Using a printer',
            'Using a smartphone', 'Walking', 'Waving', 'Writing', 'meeting'
        ]

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Convert BGR -> RGB and resize to 64x64
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        # roll sequence left, place latest at the end
        self.seq_imgs = np.roll(self.seq_imgs, shift=-1, axis=1)
        self.seq_imgs[0, -1] = img
        return self.seq_imgs

    def predict(self, frame_bgr: np.ndarray) -> str:
        # (removed extra unused 'category' arg)
        imgs = self.preprocess(frame_bgr)
        y_pred = self.model.predict(imgs, verbose=False)[0]
        argmax = np.argmax(y_pred)
        confident_score = y_pred[argmax]
        class_name = self.class_names[argmax] if confident_score > self.threshold else "No action detected"
        return class_name
