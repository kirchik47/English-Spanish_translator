import numpy as np
from tensorflow import keras
import tensorflow as tf
import pathlib


np.random.seed(42)
@keras.saving.register_keras_serializable()
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_length, embed_size, dtype=tf.float64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.max_length = max_length
        self.embed_size = embed_size
        P_matrix = np.empty(shape=(1, max_length, embed_size))
        p, i = np.meshgrid(np.arange(max_length), 2 * np.arange(embed_size // 2))
        # print(p, i)
        P_matrix[0, :, ::2] = np.sin(p / 10000**(i / embed_size)).T
        P_matrix[0, :, 1::2] = np.cos(p / 10000**((i - 1) / embed_size)).T
        self.pos_encodings = tf.constant(P_matrix)
        self.supports_masking = True
        # print(self.pos_encodings)

    def call(self, inputs):
        # print(inputs.shape)
        batch_max_len = tf.shape(inputs)[1]
        return inputs + self.pos_encodings[:batch_max_len]


def translate(sentence_en):
    translation = ""
    for word_idx in range(max_length):
        X = np.array([sentence_en])
        X_dec = np.array(["startofseq" + translation])
        y_proba = model.predict((X, X_dec))[0, word_idx]
        predicted_word_id = np.argmax(y_proba)
        predicted_word = es_vectorizer.get_vocabulary()[predicted_word_id]
        if predicted_word == 'endofseq':
            break
        translation += " " + predicted_word
    return translation

url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
path = keras.utils.get_file('spa_eng.zip', origin=url, extract=True, cache_dir='nmt_datasets')
print(path)
text = (pathlib.Path(path).with_name("spa-eng") / "spa.txt").read_text(encoding='utf-8')
text = text.replace("¡", "").replace("¿", "")
# print([line.split('\t') for line in text.splitlines()])
pairs = [line.split('\t') for line in text.splitlines()]
np.random.shuffle(pairs)
sentences_en, sentences_es = zip(*pairs)
print(sentences_en)
vocab_size = 10000
max_length = 50
embed_size = 128
# print(sentences_es)
en_vectorizer = keras.layers.TextVectorization(max_tokens=vocab_size, output_sequence_length=max_length)
es_vectorizer = keras.layers.TextVectorization(max_tokens=vocab_size, output_sequence_length=max_length)
en_vectorizer.adapt(sentences_en)
es_vectorizer.adapt([f"startofseq {s} endofseq" for s in sentences_es])

X_train = tf.constant(sentences_en[:100_000])
X_valid = tf.constant(sentences_en[100_000:])
print(X_train)
X_train_dec = tf.constant([f"startofseq {s}" for s in sentences_es[:100_000]])
X_valid_dec = tf.constant([f"startofseq {s}" for s in sentences_es[100_000:]])
y_train = es_vectorizer([f"{s} endofseq" for s in sentences_es[:100_000]])
y_valid = es_vectorizer([f"{s} endofseq" for s in sentences_es[100_000:]])


encoder_inputs = keras.layers.Input(shape=[], dtype=tf.string)
decoder_inputs = keras.layers.Input(shape=[], dtype=tf.string)
encoder_inputs_ids = en_vectorizer(encoder_inputs)
decoder_inputs_ids = es_vectorizer(decoder_inputs)

encoder_embedding_layer = keras.layers.Embedding(vocab_size, output_dim=embed_size, mask_zero=True)
decoder_embedding_layer = keras.layers.Embedding(vocab_size, output_dim=embed_size, mask_zero=True)
pos_encoding = PositionalEncoding(max_length, embed_size)
# pos_embed = keras.layers.Embedding(max_length, embed_size)
encoder_embeddings = encoder_embedding_layer(encoder_inputs_ids)
batch_max_len_enc = tf.shape(encoder_embeddings)[1] 
encoder_in = pos_encoding(encoder_embeddings)
# print(tf.shape(encoder_embeddings))

N = 6
num_heads = 8
n_units = 128
dropout_rate= 0.2
encoding_mask = tf.math.not_equal(encoder_inputs_ids, 0)[:, tf.newaxis]
Z = encoder_in
for _ in range(N):
    skip = Z
    attn_layer = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_size, dropout=dropout_rate)
    Z = attn_layer(Z, value=Z, attention_mask=encoding_mask)
    Z = keras.layers.LayerNormalization()(keras.layers.Add()([Z, skip]))
    skip = Z
    Z = keras.layers.Dense(n_units, activation='relu')(Z)
    Z = keras.layers.Dense(embed_size)(Z)
    Z = keras.layers.Dropout(rate=dropout_rate)(Z)
    Z = keras.layers.LayerNormalization()(keras.layers.Add()([Z, skip]))


decoder_embeddings = decoder_embedding_layer(decoder_inputs_ids)
batch_max_len_dec = tf.shape(decoder_embeddings)[1]

decoder_in = pos_encoding(decoder_embeddings)


decoding_mask = tf.math.not_equal(decoder_inputs_ids, 0)[:, tf.newaxis]
decoding_causal_mask = tf.linalg.band_part(tf.ones(shape=(batch_max_len_dec, batch_max_len_dec), dtype=tf.bool), -1, 0)
# print(decoding_causal_mask.numpy())
encoder_outputs = Z
Z = decoder_in
for _ in range(N):
    skip = Z
    attn_layer = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_size, dropout=dropout_rate)
    Z = attn_layer(Z, value=Z, attention_mask=decoding_mask & decoding_causal_mask)
    Z = keras.layers.LayerNormalization()(keras.layers.Add()([Z, skip]))
    skip = Z
    attn_layer = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_size, dropout=dropout_rate)
    Z = attn_layer(Z, value=encoder_outputs, attention_mask=encoding_mask)
    Z = keras.layers.LayerNormalization()(keras.layers.Add()([Z, skip]))
    skip = Z
    Z = keras.layers.Dense(n_units, activation='relu')(Z)
    Z = keras.layers.Dense(embed_size)(Z)
    Z = keras.layers.LayerNormalization()(keras.layers.Add()([Z, skip]))

# encoder = keras.layers.Bidirectional(keras.layers.LSTM(512, return_state=True, return_sequences=True))
# encoder_outputs, *encoder_state= encoder(encoder_in)
# encoder_state = [tf.concat(encoder_state[::2], axis=-1), tf.concat(encoder_state[1::2], axis=-1)]

# attention_layer = keras.layers.Attention()

# decoder = keras.layers.LSTM(1024, return_sequences=True)
# decoder_outputs = decoder(decoder_in, initial_state=encoder_state)

# attention_outputs = attention_layer([decoder_outputs, encoder_outputs])

output_layer = keras.layers.Dense(vocab_size, activation='softmax')
y_proba = output_layer(Z)


model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=y_proba)
callbacks = [keras.callbacks.ModelCheckpoint('nmt_model_N=6_dropout=0.2_heads=8_vocab_size=10000.tf', save_weights_only=True, save_format='tf', encoding='utf-8')]
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
model.load_weights('nmt_model_N=6_dropout=0.2_heads=8_vocab_size=10000.tf')
# predictions = tf.argmax(model.predict((X_valid[:1], X_valid_dec[:1])), axis=-1)
# first_3_index = tf.where(predictions == 3)[0][1]
# predictions = tf.concat([predictions[:first_3_index], tf.zeros_like(predictions[first_3_index+1:])], axis=-1)
# print(predictions)
# print(y_valid[:1])
# print(keras.metrics.Precision()(tf.cast(y_valid[:1], tf.float32), tf.cast(predictions, tf.float32)))
print(model.summary())
# model.evaluate((X_valid, X_valid_dec), y_valid)
model.fit((X_train, X_train_dec), y_train, validation_data=((X_valid, X_valid_dec), y_valid), epochs=10, batch_size=128, callbacks=callbacks)
# model.save('nmt_model.keras')
print(translate("I like playing soccer and basketball in the evening"))
