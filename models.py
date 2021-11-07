import random

import tensorflow as tf


def load_random_model() -> (tf.keras.Model, str, int):
    models = [
        [tf.keras.applications.DenseNet121, 'DenseNet121', 224],
        [tf.keras.applications.DenseNet169, 'DenseNet169', 224],
        [tf.keras.applications.DenseNet201, 'DenseNet201', 224],
        [tf.keras.applications.EfficientNetB0, 'EfficientNetB0', 224],
        [tf.keras.applications.EfficientNetB1, 'EfficientNetB1', 224],
        [tf.keras.applications.EfficientNetB2, 'EfficientNetB2', 224],
        [tf.keras.applications.EfficientNetB3, 'EfficientNetB3', 224],
        [tf.keras.applications.EfficientNetB4, 'EfficientNetB4', 224],
        [tf.keras.applications.EfficientNetB5, 'EfficientNetB5', 224],
        [tf.keras.applications.EfficientNetB6, 'EfficientNetB6', 224],
        [tf.keras.applications.EfficientNetB7, 'EfficientNetB7', 224],
        [tf.keras.applications.InceptionResNetV2, 'InceptionResNetV2', 299],
        [tf.keras.applications.InceptionV3, 'InceptionV3', 299],
        [tf.keras.applications.MobileNet, 'MobileNet', 224],
        [tf.keras.applications.MobileNetV2, 'MobileNetV2', 224],
        [tf.keras.applications.MobileNetV3Large, 'MobileNetV3Large', 224],
        [tf.keras.applications.MobileNetV3Small, 'MobileNetV3Small', 224],
        [tf.keras.applications.NASNetLarge, 'NASNetLarge', 331],
        [tf.keras.applications.NASNetMobile, 'NASNetMobile', 224],
        [tf.keras.applications.ResNet101, 'ResNet101', 224],
        [tf.keras.applications.ResNet101V2, 'ResNet101V2', 224],
        [tf.keras.applications.ResNet152, 'ResNet152', 224],
        [tf.keras.applications.ResNet152V2, 'ResNet152V2', 224],
        [tf.keras.applications.ResNet50, 'ResNet50', 224],
        [tf.keras.applications.ResNet50V2, 'ResNet50V2', 224],
        [tf.keras.applications.VGG16, 'VGG16', 224],
        [tf.keras.applications.VGG19, 'VGG19', 224],
    ]
    model_func, model_name, resolution = random.choice(models)
    model: tf.keras.Model = model_func()
    return model, model_name, resolution
