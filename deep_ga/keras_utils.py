import json
from tensorflow import keras
import tensorflow as tf
import keras.backend as K
from layers import custom_objects
from deepdiff import DeepDiff


def get_all_layers(model):
    if not hasattr(model, "layers"):
        return model
    layers = []
    for layer in model.layers:
        if hasattr(layer, "layers"):
            layers.extend(get_all_layers(layer))
        else:
            layers.append(layer)
    return layers


def count_weights(model):
    return sum(K.count_params(w) for w in model.weights)


def try_copying_weights(model1, model2):
    for layer2 in get_all_layers(model2):
        if hasattr(layer2, "layers"):
        s = count_weights2
        if s == 0:
            continue
        for layer in get_all_layers(model1):
             if sum(K.count_params(w) for w in layer.weights) == s:
                  print(layer.name, layer2.name)
                   try:
                        layer.set_weights(layer2.get_weights())
                    except:
                        print("failed to copy")

def find_equal_model(model, folder, name):
    i=1
    filepath=os.path.join(folder,f"{name}_{i:02}.hdf5")
    while os.path.isfile(filepath):
        model2=keras.models.load_model(
            filepath,
            custom_objects = custom_objects,
            compile=False
        )
        json1=json.loads(re.sub(r"_\d+", "", model.to_json()))
        json2=json.loads(re.sub(r"_\d+", "", model2.to_json()))
        diff=DeepDiff(json2,json1, exclude_regex_paths=r"\['function'\]\[0\]")
        if len(diff)==0:
            print(f"Found file with same model structure: {filepath}")
            model=model2
            # change suffix
            for layer in model.layers:
                if "mobileNetV2" in layer.name:
                    suffix=layer.name[len("mobileNetV2_"):]
                    for w in layer.weights:
                    split_name = w.name.split('/')
                    w._handle_name = split_name[0] + suffix + '/' + split_name[1] + suffix
            return model
        else:
            print(f"Found file but has different model structure: {filepath}")
        i=i+1
        filepath=os.path.join(folder,f"{name}_{i:02}.hdf5")
       
