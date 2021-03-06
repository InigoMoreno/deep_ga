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
                break


class PlotPrediction(keras.callbacks.Callback):
    def __init__(self, folder, tgen, vgen, save=True):
        self.folder = folder
        self.tgen = tgen
        self.vgen = vgen
        self.save = save
        if self.save:
            os.makedirs(folder, exist_ok=True)
        super(PlotPrediction, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        ylabel = ("Match Probability"
                  if self.model.loss.__name__ == "BCE"
                  else "Predicted distance")
        tx, ty = self.tgen.get_batch(200)
        typ = self.model.predict(tx)[:, 0]
        plt.clf()
        plt.scatter(ty, typ, 2)
        plt.title("Training Data")
        plt.xlabel("Distance between local and global patch")
        plt.xlim(0, plt.xlim()[1])
        plt.ylim(0, plt.ylim()[1])
        if self.model.loss.__name__ == "BCE":
            plt.ylim(0, 1)
        plt.ylabel(ylabel)
        if self.save:
            plt.savefig(os.path.join(self.folder, f"train_{epoch:03}.pdf"))
        else:
            plt.show()

        vx, vy = self.vgen.get_batch(200)
        vyp = self.model.predict(vx)[:, 0]
        plt.clf()
        plt.scatter(vy, vyp, 2)
        plt.title("Validation Data")
        plt.xlabel("Distance between local and global patch")
        plt.xlim(0, plt.xlim()[1])
        plt.ylim(0, plt.ylim()[1])
        if self.model.loss.__name__ == "BCE":
            plt.ylim(0, 1)
        plt.ylabel(ylabel)
        if self.save:
            plt.savefig(os.path.join(self.folder, f"valid_{epoch:03}v.pdf"))
        else:
            plt.show()


class ValidationProgbar(keras.callbacks.Callback):
    def __init__(self, vgen):
        self.pbar = None
        self.N = len(vgen)
        super(ValidationProgbar, self).__init__()

    def on_test_batch_end(self, batch, logs=None):
        if batch == 0:
            print("\nValidation:")
            self.pbar = keras.utils.Progbar(
                self.N, stateful_metrics=logs.keys())
        self.pbar.add(1, values=logs.items())
