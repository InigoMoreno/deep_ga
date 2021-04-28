import deep_ga
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt


class AssertWeightsFinite(keras.callbacks.Callback):
    def __init__(self):
        super(AssertWeightsFinite, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        for weight in self.model.weights:
            if not tf.math.reduce_all(tf.math.is_finite(weight)):
                print(f"Weight {weight.name} not finite, stop training")
                self.model.stop_training = True


class PlotPrediction(keras.callbacks.Callback):
    def __init__(self, folder, tgen, vgen):
        self.folder = folder
        self.tgen = tgen
        self.vgen = vgen
        os.makedirs(folder, exist_ok=True)
        super(PlotPrediction, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        tx, ty = self.tgen.get_batch(200)
        typ = self.model.predict(tx)[:, 0]
        plt.clf()
        plt.scatter(ty, typ, 2)
        plt.xlabel("True distance")
        plt.ylabel("Predicted distance")
        plt.savefig(os.path.join(self.folder, f"train_{epoch:03}.pdf"))

        vx, vy = self.vgen.get_batch(200)
        vyp = self.model.predict(vx)[:, 0]
        plt.clf()
        plt.scatter(vy, vyp, 2)
        plt.xlabel("True distance")
        plt.ylabel("Predicted distance")
        plt.savefig(os.path.join(self.folder, f"valid_{epoch:03}v.pdf"))
