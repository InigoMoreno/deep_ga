import json
from tensorflow import keras
import tensorflow as tf
import keras.backend as K
from deep_ga import custom_objects
from deepdiff import DeepDiff
import os


def get_all_layers(model, exclude="mobileNet"):
    if not hasattr(model, "layers"):
        return model
    layers = []
    for layer in model.layers:
        if hasattr(layer, "layers") and (exclude is None or exclude not in layer.name):
            layers.extend(get_all_layers(layer))
        else:
            layers.append(layer)
    return layers


def count_weights(model):
    return sum(K.count_params(w) for w in model.weights)


def try_copying_weights(model1, model2):
    for layer2 in get_all_layers(model2):
        s = count_weights(layer2)
        if s == 0:
            continue
        for layer in get_all_layers(model1):
            if count_weights(layer) == s:
                print(layer.name, layer2.name)
                try:
                    layer.set_weights(layer2.get_weights())
                except:
                    print("failed to copy")


def are_models_equal(model1, model2):
    json1 = json.loads(re.sub(r"_\d+", "", model1.to_json()))
    json2 = json.loads(re.sub(r"_\d+", "", model2.to_json()))
    diff = DeepDiff(
        json2, json1, exclude_regex_paths=r"\['function'\]\[0\]")
    return len(diff) == 0


def find_equal_model(model, folder):
    for filename in filter(lambda f: f.endswith(".hdf5"), os.listdir(folder)):
        filepath = os.path.join(folder, filename)
        model2 = keras.models.load_model(
            os.path.join(folder,),
            custom_objects=custom_objects,
            compile=False
        )
        if are_models_equal(model, model2):
            return filepath, modell2
