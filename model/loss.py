import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy,categorical_crossentropy

smooth = 0.001

def dice_loss_inner_layers(y_true, y_pred):
    n_classes = y_true[-1]
    #for layer in range(n_classes):
    y_true_f = K.flatten(y_true[:,:,:,1:])
    y_pred_f = K.flatten(y_pred[:,:,:,1:])
    intersect = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f + y_pred_f)
    dice_coef = 2*(intersect+smooth) / (denom+smooth)
    return 1-dice_coef

# for domain adaption: 9, else 5
def weighted_CE(y_true, y_pred, weight=tf.constant([[1.0, 1.0, 3.5, 1.0]])):
#def weighted_CE(y_true, y_pred, weight=tf.constant([[1.0, 1.0, 1.0, 1.0]])):
    print(weight)
    y_true_weighted = y_true * weight
    return categorical_crossentropy(y_true_weighted, y_pred)

# define multiclass dice coef
def dice_coef_multiclass(y_true, y_pred):
    n_classes = y_true[-1]
    #for layer in range(n_classes):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f + y_pred_f)
    dice_coef = 2*(intersect+smooth) / (denom+smooth)
    dice_loss = 1-dice_coef
    return dice_loss



# define multiclass dice coef
def dice_loss_multiclass_one_hot(y_true, y_pred):
    n_classes = y_true.shape[-1]
    y_pred_f = K.one_hot(K.cast(K.argmax(y_true, axis=-1), dtype=tf.int32), 4)
    #for layer in range(n_classes):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f + y_pred_f)
    dice_coef = 2*(intersect+smooth) / (denom+smooth)
    dice_loss = 1-dice_coef
    return K.cast(dice_loss, dtype=tf.float32)


def dice_coef_multiclass_avg(y_true, y_pred, n_classes=4):
    dice_coef_classes = []
    for idx in range(n_classes):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersect = K.sum(y_true_f * y_pred_f)
        denom = K.sum(y_true_f + y_pred_f)
        dice_coef = 2*(intersect+smooth) / (denom+smooth)
        dice_coef_classes.append(dice_coef)
    return tf.math.reduce_mean(dice_coef_classes)

def dice_coef_multiclass_avg(y_true, y_pred, n_classes=4):
    dice_coef_classes = []
    for idx in range(n_classes):
        y_true_f = K.flatten(y_true[:,:,:,idx])
        y_pred_f = K.flatten(y_pred[:,:,:,idx])
        intersect = K.sum(y_true_f * y_pred_f)
        denom = K.sum(y_true_f + y_pred_f)
        dice_coef = 2*(intersect+smooth) / (denom+smooth)
        dice_coef_classes.append(dice_coef)
    return tf.math.reduce_mean(dice_coef_classes) #dice_coef_classes

def dice_coef_avg_loss(y_true, y_pred):
    return 1-dice_coef_multiclass_avg(y_true, y_pred)

def weighted_CE_dice(y_true, y_pred):
    CE = weighted_CE(y_true, y_pred)
    return (dice_loss_multiclass_one_hot(y_true, y_pred)) + CE

def CE_dice(y_true, y_pred):
    CE = categorical_crossentropy(y_true, y_pred)
    return (dice_loss_multiclass_one_hot(y_true, y_pred)) + CE
    
def dice_coef_multiclass_with_categorical_CE(y_true, y_pred):
    CE = categorical_crossentropy(y_true, y_pred)
    # return -dice_coef(y_true, y_pred)+CE
    return dice_coef_multiclass_loss(y_true, y_pred) + CE

# Metric function
def dice_coef(y_true, y_pred):
    # y_true /= 255.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multiclass_one_hot(y_true, y_pred):
    n_classes = y_true.shape[-1]
    y_pred_f = K.one_hot(K.cast(K.argmax(y_true, axis=-1), dtype=tf.int32), 4)
    #for layer in range(n_classes):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f + y_pred_f)
    dice_coef = 2*(intersect+smooth) / (denom+smooth)
    return K.cast(dice_coef, dtype=tf.float32)


# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def dice_coef_loss_with_CE(y_true, y_pred):
    CE = binary_crossentropy(y_true, y_pred)
    return -dice_coef(y_true, y_pred) + CE


def weighted_dice_with_CE(y_true, y_pred):
    CE = binary_crossentropy(y_true, y_pred)
    # return -dice_coef(y_true, y_pred)+CE
    return 0.2 * (1 - dice_coef(y_true, y_pred)) + CE

def weighted_dice_with_categorical_CE(y_true, y_pred):
    CE = categorical_crossentropy(y_true, y_pred)
    # return -dice_coef(y_true, y_pred)+CE
    return 0.2 * (dice_loss_multiclass_one_hot(y_true, y_pred)) + CE

# Tversky Metric function
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return 1.0 - K.pow((1 - pt_1), gamma)


def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def focal_loss(gamma=2., alpha=0.25):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed