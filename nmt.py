import numpy as np
import pathlib
from collections import Counter
import pandas as pd
import os
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers.keras_callbacks import PushToHubCallback
# from huggingface_hub import login, HfApi, push_to_hub_keras
# from tensorflow import keras
import tensorflow as tf
import keras


# login(token=os.getenv('HF_TOKEN'))

cur_dir = pathlib.Path(__file__).parent.resolve()
filepath = 'nmt_model_N=4_dropout=0.2_heads=8_vocab_size=10002_BPE.keras'
    
np.random.seed(42)
tf.random.set_seed(42)

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


class Tokenizer(keras.layers.Layer):
    def __init__(self, merges, max_length, **kwargs):
        super().__init__(**kwargs)
        self.merges = merges
        self.max_length = max_length
    
    def encode(self, sentence, eos):
        j = 0
        while j < min(len(sentence), max_length):
            if j + 1 < len(sentence):
                idx = self.merges.get((sentence[j], sentence[j + 1]), 0)
                if idx:
                    sentence.pop(j)
                    sentence[j] = idx
                else:
                    j += 1
            else:
                j += 1
        if len(sentence) < self.max_length:
            if eos:
                sentence.append(10001)
                j += 1
            while j < self.max_length:
                sentence.append(0)
                j += 1
        else:
            while len(sentence) != self.max_length:
                sentence.pop()
            if eos:
                sentence[-1] = 10001
        
        return sentence
    
    def decode(self, sentence):
        j = 0
        values = list(self.merges.values())
        while j < len(sentence):
            idx = sentence[j]
            if idx in values:
                # print(sentence)
                sentence[j] = list(self.merges.keys())[values.index(idx)][1]
                # print(sentence)
                sentence.insert(j, list(self.merges.keys())[values.index(idx)][0])
                # print(sentence)
            else:
                j += 1
        print(sentence)
        if sentence[-1] == 10001:
            return bytes(sentence[:-1]).decode() + 'endofseq'
        return bytes(sentence).decode()

    
    def call(self, inputs, encode=True, sos=False, eos=False):
        tokenized_inputs = []
        try:
            inputs = inputs.numpy()
        except:
            pass
        # print(inputs)
        if (len(inputs.shape) > 1 and encode) or (len(inputs.shape) > 2 and not encode):
            for batch in inputs:
                for sentence in batch:
                    sentence = list(sentence)
                    if encode:
                        sentence = self.encode(sentence)
                    else:
                        sentence = self.decode(sentence)
                    tokenized_inputs.append(sentence)
        else:
            for sentence in inputs:
                sentence = list(sentence)
                # print(sentence)
                if encode:
                    sentence = self.encode(sentence, eos)
                else:
                    sentence = self.decode(sentence)
                if sos:
                    sentence = [10000] + sentence
                    sentence.pop()
                tokenized_inputs.append(sentence)
                    
            # print(tokenized_inputs)
        if encode:
            return tf.constant(tokenized_inputs)
        return tokenized_inputs

@keras.saving.register_keras_serializable()
class CHRF(keras.metrics.Metric):
    def __init__(self, n_grams=3, name='chrf', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_grams = n_grams
        self.chrf_score = self.add_weight(name='chrf_score', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = keras.ops.argmax(y_pred, axis=-1)
        indices = keras.ops.concatenate([keras.ops.arange(i, i + y_true.shape[1] - self.n_grams + 1)[:, tf.newaxis] for i in range(self.n_grams)], axis=-1)
        references = keras.ops.concatenate([keras.ops.take(y_true, indices[i], axis=-1) for i in range(indices.shape[0])], axis=0)
        indices = keras.ops.concatenate([keras.ops.arange(i, i + y_pred.shape[1] - self.n_grams + 1)[:, tf.newaxis] for i in range(self.n_grams)], axis=-1)
        predictions = keras.ops.concatenate([keras.ops.take(y_pred, indices[i], axis=-1) for i in range(indices.shape[0])], axis=0)
        references = tf.strings.reduce_join(tf.as_string(references), axis=-1)
        predictions = tf.strings.reduce_join(tf.as_string(predictions), axis=-1)
        # print(tf.equal(predictions, references))
        # references, _ = tf.unique(references)
        # predictions, _ = tf.unique(predictions)
        num_common_ngrams = keras.ops.cast(keras.ops.sum(keras.ops.equal(predictions, references)), tf.float64)
        # common_ngrams = tf.sets.intersection([references], [predictions])
        # num_common_ngrams = tf.shape(common_ngrams.values)[0]
        precision = keras.ops.cond(keras.ops.shape(predictions)[0] > 0, 
                        lambda: num_common_ngrams / keras.ops.cast(keras.ops.shape(predictions)[0], tf.float64), 
                        lambda: keras.ops.cast(0.0, tf.float64))
        recall = keras.ops.cond(keras.ops.shape(references)[0] > 0, 
                     lambda: num_common_ngrams / keras.ops.cast(keras.ops.shape(references)[0], tf.float64), 
                     lambda: keras.ops.cast(0.0, tf.float64))
        f1_score = keras.ops.cond(precision + recall > 0, 
                   lambda: 2 * (precision * recall) / (precision + recall), 
                   lambda: keras.ops.cast(0.0, tf.float64))
        f1_score = keras.ops.cast(f1_score, tf.float32)
        self.chrf_score.assign_add(f1_score)
        self.count.assign_add(1)

    def result(self):
        return self.chrf_score / self.count
    
    def reset_state(self):
        self.chrf_score.assign(0)
        self.count.assign(0)

    def get_config(self, **kwargs):
        return super().get_config(**kwargs)
    
    
def get_dir():
    log_dir = os.path.join(cur_dir, 'nmt_logs')
    run_dir = os.path.join(log_dir, time.strftime('run_%Y-%m-%d %H-%M-%S') + filepath)
    return run_dir

def translate(sentence_en, model_path):
    en_tokenizer = Tokenizer(merges_en, max_length)
    es_tokenizer = Tokenizer(merges_es, max_length)
    model = keras.models.load_model(model_path, custom_objects={'CHRF': CHRF})
    translation = ""
    X = en_tokenizer(tf.constant([sentence_en]))

    for word_idx in range(max_length):
        X_dec = es_tokenizer(tf.constant([translation]), sos=True)
        y_proba = model.predict((X, X_dec))[0, word_idx]
        predicted_word_id = np.argmax(y_proba)
        print(word_idx)
        predicted_word = es_tokenizer(tf.constant([[predicted_word_id]]), encode=False)[0]
        print(translation)
        if predicted_word == 'endofseq':
            break
        translation += predicted_word
    return translation


def get_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
    path = keras.utils.get_file('spa_eng.zip', origin=url, extract=True, cache_dir='nmt_datasets')
    print(path)
    text = (pathlib.Path(path).with_name("spa-eng") / "spa.txt").read_text(encoding='utf-8')
    text = text.replace("¡", "").replace("¿", "")
    pairs = [line.split('\t') for line in text.splitlines()]
    np.random.shuffle(pairs)
    sentences_en, sentences_es = zip(*pairs)
    return sentences_en, sentences_es

def build_model():
    encoder_inputs_ids = keras.Input(shape=[max_length], dtype=tf.int64)
    decoder_inputs_ids = keras.Input(shape=[max_length], dtype=tf.int64)

    encoder_embedding_layer = keras.layers.Embedding(vocab_size, output_dim=embed_size, mask_zero=True)
    decoder_embedding_layer = keras.layers.Embedding(vocab_size, output_dim=embed_size, mask_zero=True)
    pos_encoding = PositionalEncoding(max_length, embed_size)
    # pos_embed = keras.layers.Embedding(max_length, embed_size)
    encoder_embeddings = encoder_embedding_layer(encoder_inputs_ids)
    encoder_in = pos_encoding(encoder_embeddings)
    # print(tf.shape(encoder_in))

    N = 4
    num_heads = 8
    n_units = 128
    dropout_rate= 0.2
    encoding_mask = keras.ops.not_equal(encoder_inputs_ids, 0)[:, tf.newaxis]
    # print(tf.shape(encoding_mask))
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
    decoder_in = pos_encoding(decoder_embeddings)

    decoding_mask = keras.ops.not_equal(decoder_inputs_ids, 0)[:, tf.newaxis]
    batch_max_len = keras.ops.shape(decoder_embeddings)[1]
    decoding_causal_mask = keras.ops.tri(batch_max_len, batch_max_len, 0, dtype=tf.bool)

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

    output_layer = keras.layers.Dense(vocab_size, activation='softmax')
    y_proba = output_layer(Z)


    model = keras.models.Model(inputs=[encoder_inputs_ids, decoder_inputs_ids], outputs=y_proba)
    return model


def train():
    en_tokenizer = Tokenizer(merges_en, max_length)
    es_tokenizer = Tokenizer(merges_es, max_length)

    # temp = es_tokenizer(tf.constant(['endofseq', 'govno']), encode=True)
    # print(es_tokenizer(temp, encode=False), temp)
    X_train = en_tokenizer(tf.constant(sentences_en[:100_000]))
    X_valid = en_tokenizer(tf.constant(sentences_en[100_000:]))
    # print(X_train)
    X_train_dec = es_tokenizer(tf.constant(sentences_es[:100_000]), sos=True)
    print(X_train_dec)
    X_valid_dec = es_tokenizer(tf.constant(sentences_es[100_000:]), sos=True)
    y_train = es_tokenizer(tf.constant(sentences_es[:100_000]), eos=True)
    y_valid = es_tokenizer(tf.constant(sentences_es[100_000:]), eos=True)

    model = build_model()
    # model = keras.models.load_model(filepath=filepath, custom_objects={'CHRF': CHRF}) 
    
    callbacks = [keras.callbacks.ModelCheckpoint(filepath, save_best_only=True), 
                 keras.callbacks.ReduceLROnPlateau(),
                 keras.callbacks.TensorBoard(log_dir=get_dir())]
    print(model.summary())
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=["accuracy", CHRF(3)])
    # model.evaluate((X_valid, X_valid_dec), y_valid)
    model.fit((X_train, X_train_dec), y_train, validation_data=((X_valid, X_valid_dec), y_valid), epochs=3, batch_size=128, callbacks=callbacks)
    # push_to_hub_keras(model, 'kirchik47/english-spanish-translator')
    return model


if __name__ == '__main__':
    sentences_en, sentences_es = get_data()
    print(sentences_en)

    vocab_size = 10002 # 2 for sos and eos
    max_length = 50
    embed_size = 128

    merges_en = pd.read_csv('merges_en.csv')
    merges_en = {key: value for key, value in zip(zip(merges_en['pair0'], merges_en['pair1']), merges_en['idx'])}
    
    merges_es = pd.read_csv('merges_es.csv')
    merges_es = {key: value for key, value in zip(zip(merges_es['pair0'], merges_es['pair1']), merges_es['idx'])}
    print(merges_es)
    train()
    print(translate("They're sunbathing around the pool.", filepath))
