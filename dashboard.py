import io

import streamlit as st
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import librosa
import matplotlib.pyplot as plt


def load_image_classification_model():
    base_model = keras.applications.EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(6, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.load_weights('Image_class_weights/EfficientNetV2S_checkpoint.weights.h5')

    return model


def load_audio_classification_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(42,)),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(10, activation='softmax')
    ])

    model.load_weights('audio_class_weights/FFNN_Fullfeatures.weights.h5')

    return model


def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0


def display_generated_samples(example_sample, model):
    generated_sample = model.predict(example_sample)
    return generated_sample


conv_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


def encoder_block(input_layer, filters, size=3, strides=2, apply_instancenorm=True, activation=keras.layers.ReLU()):
    block = keras.layers.Conv2D(filters, size,
                                strides=strides,
                                padding='same',
                                use_bias=False,
                                kernel_initializer=conv_initializer)(input_layer)

    if apply_instancenorm:
        block = keras.layers.LayerNormalization(gamma_initializer=gamma_initializer)(block)

    block = activation(block)

    return block


def transformer_block(input_layer, size=3, strides=1):
    filters = input_layer.shape[-1]

    block = keras.layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=False,
                                kernel_initializer=conv_initializer)(input_layer)

    block = keras.layers.ReLU()(block)

    block = keras.layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=False,
                                kernel_initializer=conv_initializer)(block)

    block = keras.layers.Add()([block, input_layer])

    return block


def decoder_block(input_layer, filters, size=3, strides=2):
    block = keras.layers.Conv2DTranspose(filters, size,
                                         strides=strides,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=conv_initializer)(input_layer)

    block = keras.layers.LayerNormalization(gamma_initializer=gamma_initializer)(block)

    block = keras.layers.ReLU()(block)

    return block


def generator_fn(height=256, width=256, channels=3):
    OUTPUT_CHANNELS = 3
    inputs = keras.layers.Input(shape=[height, width, channels])

    # Encoder
    enc_1 = encoder_block(inputs, 64, 7, 1, apply_instancenorm=False)
    enc_2 = encoder_block(enc_1, 128, 3, 2, apply_instancenorm=True)
    enc_3 = encoder_block(enc_2, 256, 3, 2, apply_instancenorm=True)

    # Transformer
    x = enc_3
    for n in range(6):
        x = transformer_block(x, 3, 1)

    # Decoder
    x_skip = keras.layers.Concatenate()([x, enc_3])

    dec_1 = decoder_block(x_skip, 128, 3, 2)
    x_skip = keras.layers.Concatenate()([dec_1, enc_2])

    dec_2 = decoder_block(x_skip, 64, 3, 2)
    x_skip = keras.layers.Concatenate()([dec_2, enc_1])

    outputs = last = keras.layers.Conv2D(OUTPUT_CHANNELS, 7,
                                         strides=1, padding='same',
                                         kernel_initializer=conv_initializer,
                                         use_bias=False,
                                         activation='tanh')(x_skip)

    generator = keras.Model(inputs, outputs)

    return generator


def load_gan_model():
    model = generator_fn(height=256, width=256)
    model.load_weights('gan_weights/monet_generator.weights.h5')
    return model


def main():
    st.set_page_config(page_title="DSIM project", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page:", ["Audio Classification", "Image Classification", "Style Transfer"])

    if page == "Audio Classification":
        audio_classification_page()
    elif page == "Image Classification":
        image_classification_page()
    elif page == "Style Transfer":
        style_transfer_page()


# Page 1: Audio Classification
def audio_classification_page():
    st.title("Audio Classification")
    st.write("Upload an audio file to classify it.")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        st.success("Audio uploaded successfully!")

        class_names = {0: "blues",
                       1: "classical",
                       2: "country",
                       3: "disco",
                       4: "hiphop",
                       5: "jazz",
                       6: "metal",
                       7: "pop",
                       8: "raggae",
                       9: "rock"}

        y, sr = librosa.load(io.BytesIO(uploaded_file.read()), sr=None)

        mfcc = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=40)

        mfcc_feat = np.mean(mfcc.T, axis=0)
        #mfcc_feat = np.array([mfcc_feat])

        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr_feat = np.mean(zcr)
        tempo = librosa.feature.tempo(y=y)

        features = np.append(mfcc_feat, zcr_feat)
        features = np.append(features, tempo)
        features = np.array([features])
        model = load_audio_classification_model()
        y_pred = model.predict(features)
        y_pred = np.argmax(y_pred, axis=1)[0]
        pred_class_name = class_names.get(y_pred, "Unknown")

        st.header(f"Real time prediction: {pred_class_name}")


# Page 2: Image Classification
def image_classification_page():
    st.title("Image Classification")
    st.write("Upload an image to classify it.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg"])

    if uploaded_file is not None:
        class_names = {0: "buildings",
                       1: "forest",
                       2: "glacier",
                       3: "mountain",
                       4: "sea",
                       5: "street"}

        image = Image.open(uploaded_file)
        image = image.resize((224, 224))

        model = load_image_classification_model()

        image = np.expand_dims(np.array(image), axis=0)

        prob = model.predict(image)
        pred_class = np.argmax(prob, axis=-1)[0]
        pred_class_name = class_names.get(pred_class, "Unknown")

        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            st.image(uploaded_file, caption="Uploaded Image")  # Smaller image
            st.success("Image uploaded successfully!")

        with col2:
            st.header(f"Real time prediction: {pred_class_name}")


# Page 3: Style Transfer
def style_transfer_page():
    st.title("Style Transfer")
    st.write("Upload a content image to apply style transfer.")

    content_image = st.file_uploader("Upload a content image", type=["jpg"], key="content")

    if content_image is not None:
        image = Image.open(content_image)
        image = image.resize((256, 256))
        image = np.expand_dims(np.array(image), axis=0)
        image = normalize_img(image)

        model = load_gan_model()

        result = display_generated_samples(image, model)

        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            st.header('Base image')
            st.image(content_image, caption=["Content Image"], width=300)
            st.success("Images uploaded successfully!")

        with col2:
            st.header('Style transferred Image')
            st.image(result[0] * 0.5 + 0.5, width=500)


if __name__ == "__main__":
    main()
