import deep_ga
from tensorflow import keras
import keras.backend as K
import numpy as np


def single_branch(input_tensor, hyperparams, suffix=None):
    if suffix is None:
        suffix = "_" + input_tensor.name

    tensor = input_tensor
    tensor = keras.layers.Reshape(
        (input_tensor.shape[1], input_tensor.shape[2], 1))(input_tensor)

    raw = deep_ga.NanToZero()(tensor)
    mask = deep_ga.IsNanMask()(tensor)

    if hyperparams["input"] == "raw":
        tensor = keras.layers.Concatenate()([raw, raw, raw])
    elif hyperparams["input"] == "rawmask":
        tensor = keras.layers.Concatenate()([raw, raw, mask])
    elif hyperparams["input"] == "sobel":
        tensor = deep_ga.SymConv2D(3, True)(tensor)
    elif hyperparams["input"] == "fixsobel":
        tensor = deep_ga.SymConv2D(3, True, trainable=False)(tensor)
    elif hyperparams["input"] == "PConv":
        tensor = deep_ga.PConv2D(3)(tensor)
    elif hyperparams["input"] == "SymConv":
        tensor = deep_ga.SymConv2D(3)(tensor)
    elif hyperparams["input"] == "Conv":
        tensor = deep_ga.NanToZero()(tensor)
        tensor = keras.layers.Conv2D(3, 3)(tensor)
    else:
        raise ValueError(f"unknown input {hyperparams['input']}")

    if hyperparams["mobileNet_weights"] is not None:
        #sizes=[96, 128, 160, 192, 224]
        tensor = keras.layers.experimental.preprocessing.Resizing(
            96, 96)(tensor)

    # tensor = keras.applications.mobilenet_v2.preprocess_input(tensor)
    mobileNet = keras.applications.MobileNetV2(
        input_shape=tensor.shape[1:],
        alpha=hyperparams["mobileNet_alpha"],
        include_top=False,
        pooling=hyperparams["mobileNet_pooling"],
        weights=hyperparams["mobileNet_weights"]
    )

    if hyperparams["mobileNet_weights"] is not None:
        for (i, layer) in enumerate(mobileNet.layers):
            if i < len(mobileNet.layers) - 4:
                layer.trainable = False

    tensor = mobileNet(tensor)
    if not hyperparams["learnEnding"]:
        if hyperparams["firstLayerSize"] > 0:
            tensor = keras.layers.Dense(
                hyperparams["firstLayerSize"], activation=hyperparams["activation"])(tensor)

        if hyperparams["dropout"] > 0:
            tensor = keras.layers.Dropout(hyperparams["dropout"])(tensor)

        if hyperparams["secondLayerSize"] > 0:
            tensor = keras.layers.Dense(
                hyperparams["secondLayerSize"], activation=hyperparams["activation"])(tensor)

    return tensor


def set_trainable_only_last(model):
    layers = deep_ga.get_all_layers(model.get_layer("single_branch"), None)
    idx = layers.index([l for l in layers if "block" in l.name][-1])
    for layer in layers[:idx]:
        layer.trainable = False


def get_model(hyperparams, input_a, input_b=None):
    shared_weights = hyperparams["sharedWeights"]
    if shared_weights:
        if "loadBranch" in hyperparams.keys() and hyperparams["loadBranch"] is not None:
            loadModel = keras.models.load_model(
                hyperparams["loadBranch"], custom_objects=deep_ga.custom_objects, compile=False)
            branch_model = loadModel.get_layer("single_branch")
            set_trainable_only_last(branch_model)
        else:
            branch_model = keras.Model(inputs=[input_a], outputs=single_branch(
                input_a, hyperparams, suffix=""), name="single_branch")
        embedding_a = branch_model(input_a)
        embedding_b = branch_model(input_b)
    else:
        if "loadBranch" in hyperparams.keys() and hyperparams["loadBranch"] is not None:
            loadModel = keras.models.load_model(
                hyperparams["loadBranch"], custom_objects=deep_ga.custom_objects, compile=False)
            branch_model_a = loadModel.get_layer("branch_a")
            set_trainable_only_last(branch_model_a)
            branch_model_b = loadModel.get_layer("branch_b")
            set_trainable_only_last(branch_model_b)
        else:
            branch_model_a = keras.Model(inputs=[input_a], outputs=single_branch(
                input_a, hyperparams, suffix="-a"), name="branch_a")
            branch_model_b = keras.Model(inputs=[input_b], outputs=single_branch(
                input_b, hyperparams, suffix="-b"), name="branch_b")
        embedding_b = branch_model_b(input_b)
        embedding_a = branch_model_a(input_a)

    if hyperparams["learnEnding"]:
        tensor = keras.layers.Concatenate()([embedding_a, embedding_b])
        if hyperparams["firstLayerSize"] > 0:
            tensor = keras.layers.Dense(
                hyperparams["firstLayerSize"], activation=hyperparams["activation"])(tensor)

        if hyperparams["dropout"] > 0:
            tensor = keras.layers.Dropout(hyperparams["dropout"])(tensor)

        if hyperparams["secondLayerSize"] > 0:
            tensor = keras.layers.Dense(
                hyperparams["secondLayerSize"], activation=hyperparams["activation"])(tensor)
        embedding_dist = keras.layers.Dense(1, activation="sigmoid")(tensor)
    else:
        embedding_dist = deep_ga.EuclideanDistanceLayer()(
            [embedding_a, embedding_b])

    model = keras.Model(
        inputs=[input_a, input_b], outputs=embedding_dist)
    return model


def compile_model(model, distances, hyperparams):
    def normalize(loss):
        def normalize(y_true, y_pred):
            return 100 * loss(y_true, y_pred) / normal_loss
        d = K.constant(distances)
        mean_d = K.constant(np.full_like(distances, distances.mean()))
        normal_loss = loss(d, mean_d)
        normalize.__name__ = 'N{}'.format(loss.__name__)
        return normalize

    losses = {
        "MSE": keras.losses.MSE,
        "MAE": keras.losses.MAE,
        "DOOMSE": deep_ga.doomloss,
        "PCL": deep_ga.pairwise_contrastive_loss,
        "BCE": deep_ga.binary_cross_entropy,
    }

    for name, loss in losses.items():
        loss.__name__ = name

    # losses={k: normalize(v) for k,v in losses.items()}

    if hyperparams["optimizer"] == "Adam":
        optimizer = keras.optimizers.Adam(lr=hyperparams["learning_rate"])
    elif hyperparams["optimizer"] == "SGD":
        optimizer = keras.optimizers.SGD(
            learning_rate=hyperparams["learning_rate"])
    else:
        raise ValueError(f"unknown optimizer {hyperparams['optimizer']}")

    if hyperparams["learnEnding"]:
        loss = "BCE"
    else:
        loss = hyperparams["loss"]

    model.compile(
        loss=losses[loss],
        # metrics=list(losses.values()),
        optimizer=optimizer
    )

    return model
