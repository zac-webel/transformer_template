import pandas as pd
from random import randint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from random import randint


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim,mask_zero=True):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,mask_zero=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim,mask_zero=True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions



loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False,reduction='none')
def masked_loss(y_true, y_pred):
    padding_mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = loss_fn(y_true, y_pred)
    masked_loss = tf.math.multiply(loss, tf.cast(padding_mask, dtype=loss.dtype))
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(tf.cast(padding_mask, dtype=loss.dtype))



def masked_accuracy(y_true, y_pred):
    padding_mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    y_true_labels = tf.cast(y_true, dtype=tf.int64)
    correct_predictions = tf.cast(tf.equal(y_pred_labels, y_true_labels),dtype=tf.int32)
    masked_predictions = tf.math.multiply(correct_predictions, tf.cast(padding_mask, dtype=np.int32))
    return tf.reduce_sum(tf.cast(masked_predictions, dtype=tf.float32)) / tf.reduce_sum(tf.cast(padding_mask, dtype=np.float32))





def encoder(encoder_input_embeddings):
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
    masked_multi_head_attn = attention(query=encoder_input_embeddings, value=encoder_input_embeddings, use_causal_mask=False, return_attention_scores=False)
    normalized_embeddings = layers.LayerNormalization()(encoder_input_embeddings)
    residual_connection_1 = layers.Add()([normalized_embeddings, masked_multi_head_attn])
    ff = layers.Dense(ff_dim, activation='relu')(residual_connection_1)
    ff = layers.Dense(embed_dim)(ff)
    normalized_residual_1 = layers.LayerNormalization()(residual_connection_1)
    residual_connection_2 = layers.Add()([normalized_residual_1, ff])
    return residual_connection_2

def decoder(decoder_input_embeddings, encoder_output):
    self_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
    masked_self_attn = self_attention(query=decoder_input_embeddings,value=decoder_input_embeddings,use_causal_mask=True,return_attention_scores=False)
    normalized_self_embeddings = layers.LayerNormalization()(decoder_input_embeddings)
    residual_connection_1 = layers.Add()([normalized_self_embeddings, masked_self_attn])

    cross_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
    masked_cross_attn = cross_attention(query=residual_connection_1,value=encoder_output,key=encoder_output,return_attention_scores=False)
    normalized_cross_attn = layers.LayerNormalization()(residual_connection_1)
    residual_connection_2 = layers.Add()([normalized_cross_attn, masked_cross_attn])

    ff = layers.Dense(ff_dim, activation='relu')(residual_connection_2)
    ff = layers.Dense(embed_dim)(ff)
    normalized_residual_2 = layers.LayerNormalization()(residual_connection_2)
    residual_connection_3 = layers.Add()([normalized_residual_2, ff])

    return residual_connection_3

def gpt(gpt_input_embeddings):
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
    masked_multi_head_attn = attention(query=gpt_input_embeddings, value=gpt_input_embeddings, use_causal_mask=True, return_attention_scores=False)
    normalized_embeddings = layers.LayerNormalization()(gpt_input_embeddings)
    residual_connection_1 = layers.Add()([normalized_embeddings, masked_multi_head_attn])
    ff = layers.Dense(ff_dim, activation='relu')(residual_connection_1)
    ff = layers.Dense(embed_dim)(ff)
    normalized_residual_1 = layers.LayerNormalization()(residual_connection_1)
    residual_connection_2 = layers.Add()([normalized_residual_1, ff])
    return residual_connection_2




num_heads = 2
ff_dim = 128
decoder_vocab_size = len(vocabulary)
embed_dim = 32


encoder_inputs = layers.Input(shape=(max_length,))
encoder_inputs = layers.Masking(mask_value=0)(encoder_inputs)
encoder_embeddings = TokenAndPositionEmbedding(max_length, len(vocabulary), embed_dim)(encoder_inputs)

decoder_inputs = layers.Input(shape=(max_length,))
decoder_inputs = layers.Masking(mask_value=0)(decoder_inputs)
decoder_embeddings = TokenAndPositionEmbedding(max_length, len(vocabulary), embed_dim)(decoder_inputs)

encoder_1_output = encoder(encoder_embeddings)
encoder_2_output = encoder(encoder_1_output)


decoder_1_output = decoder(decoder_embeddings,encoder_2_output)
decoder_2_output = decoder(decoder_1_output,encoder_2_output)


output = layers.Dense(len(vocabulary),activation='softmax')(decoder_2_output)
model = keras.Model(inputs=[encoder_inputs,decoder_inputs],outputs = output)


optimizer = tf.keras.optimizers.Adam(
    learning_rate=2.5e-4,
    beta_1=0.9,
    beta_2=0.999)
model.compile(loss=masked_loss, metrics=masked_accuracy,  optimizer=optimizer)


import gc
gc.enable()
for i in range(100):
    batch_encoder_inputs,batch_decoder_inputs,batch_outputs = gen_batch(10_000)
    training = model.fit(x=[batch_encoder_inputs,batch_decoder_inputs],y=batch_outputs,epochs=1,batch_size=4,verbose=1)
    del batch_encoder_inputs,batch_decoder_inputs,batch_outputs
    gc.collect()




