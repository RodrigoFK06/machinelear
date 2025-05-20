import numpy as np

class SequenceRecorder:
    def __init__(self, max_length=30):
        self.max_length = max_length
        self.frames = []

    def add_frame(self, vector):
        self.frames.append(vector)
        if len(self.frames) > self.max_length:
            self.frames.pop(0)

    def get_sequence(self):
        if len(self.frames) == self.max_length:
            return np.array(self.frames)
        return None

    def reset(self):
        self.frames = []
