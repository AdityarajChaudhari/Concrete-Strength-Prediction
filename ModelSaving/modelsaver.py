import pickle
from Tuning.ModelTuner import ModelTuning


class SaveModel:

    def __init__(self):
        self.model = ModelTuning()
        self.path = "model.pkl"
        self.mode = "wb"

    def save(self):
        model = self.model.stacking()
        path = self.path
        mode = self.mode
        pickle.dump(model, open(path, mode))


s = SaveModel()
s.save()

