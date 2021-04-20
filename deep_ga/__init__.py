from deep_ga.functions import ply_to_image, get_valid_filename
from deep_ga.patch_generator import raycast_occlusion, get_patch, get_batch, PatchDataGenerator, get_batch_local_global, LocalGlobalPatchDataGenerator
from deep_ga.layers import SymConv2D, PConv2D, EuclideanDistanceLayer, NanToZero, IsNanMask
from deep_ga.constants import get_gps_references
from deep_ga.keras_utils import get_all_layers, count_weights, try_copying_weights, are_models_equal, find_equal_model
from deep_ga.mdnt import SWATS
from deep_ga.losses import set_scale, pairwise_contrastive_loss, binary_cross_entropy, doomloss

custom_objects = {
    "SymConv2D": layers.SymConv2D,
    "PConv2D": layers.PConv2D,
    "EuclideanDistanceLayer": layers.EuclideanDistanceLayer,
    "NanToZero": layers.NanToZero,
    "pairwise_contrastive_loss": losses.pairwise_contrastive_loss,
    "binary_cross_entropy": losses.binary_cross_entropy,
    "doomloss": losses.doomloss
}
