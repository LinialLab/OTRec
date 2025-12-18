# dl_model_def.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import FeatureSpace
# REMOVED: from keras.layers import ... 

MAX_TOK = 160_000
EMB_ID  = 64

@keras.utils.register_keras_serializable(package="OTRec")
def make_fs():
    return FeatureSpace(
        {
            "text": FeatureSpace.feature(
                preprocessor=keras.layers.TextVectorization(
                    max_tokens=MAX_TOK,
                    output_mode="count",
                ),
                dtype="string",
                output_mode="float",
            )
        },
        output_mode="concat",
    )


# @keras.utils.register_keras_serializable() # added to here instead of inside the model
# def build_tower(input_dim: int,EMB_ID:int=64) -> keras.Model:
#     inp = keras.Input(shape=(input_dim + EMB_ID,))
#     x   = keras.layers.LayerNormalization()(inp)
#     # x   = keras.layers.BatchNormalization()(inp)
#     ## BatchNormalization
#     x = keras.layers.Dropout(0.2)(x)
#     # x  = keras.layers.Dense(768, activation="gelu")(x)
#     # out = keras.layers.Dense(256, activation="tanh")(x)
#     # out = keras.layers.Dense(256, activation="gelu")(inp)

#     # out = keras.layers.Dense(256, activation="linear")(x) # orig, 95.9 auc
#     # out = keras.layers.Dense(256, activation="gelu")(x) # 
#     out = keras.layers.Dense(512, activation="elu")(x)
#     return keras.Model(inp, out, name="tower")

@keras.utils.register_keras_serializable()
def build_tower(input_dim: int, EMB_ID: int = 64) -> keras.Model:
    inp = keras.Input(shape=(input_dim + EMB_ID,))
    norm_x = keras.layers.LayerNormalization()(inp)

    # Path 1: The Linear Projection (Wide)
    linear_out = keras.layers.Dense(384, activation="linear")(norm_x)

    # Path 2: Non-linear capture (Optional complex interactions)
    deep = keras.layers.Dense(384, activation="elu")(norm_x)
    deep = keras.layers.LayerNormalization()(deep) # Norm inside deep block is fine
    deep = keras.layers.Dropout(0.35)(deep)
    
    deep = keras.layers.Dense(64, activation="elu")(deep)
    deep = keras.layers.Dropout(0.15)(deep)
    # # Remove the LN here if you are putting it at the end, 
    # # OR keep it if you want the deep branch specifically standardized.
    # # (Keeping it is fine/standard for a block).
    # deep = keras.layers.LayerNormalization()(deep)
    deep = keras.layers.Dense(384, activation="linear")(deep)

    # Add them (Residual style)
    out = keras.layers.Add()([linear_out, deep]) 
    # out = keras.layers.LayerNormalization(name="final_norm")(out)

    return keras.Model(inp, out, name="tower")


@keras.utils.register_keras_serializable(package="OTRec")
class TwoTowerDual(keras.Model):
    def __init__(self,
                 dise_lookup,
                 dise_emb,
                 q_fs,
                 k_fs,
                 q_tower,
                 k_tower,
                 concat_layer,
                 **kwargs):
        super().__init__(**kwargs)
        self.dise_lookup = dise_lookup
        self.dise_emb    = dise_emb
        self.q_fs        = q_fs
        self.k_fs        = k_fs
        self.q_tower     = q_tower
        self.k_tower     = k_tower
        self.concat      = concat_layer
        self.dot         = keras.layers.Dot(axes=-1, normalize=True, name="cosine")
        self.cls_head    = keras.layers.Dense(1, activation="sigmoid", 
            name="cls",
            # 1. Start with a high scaling factor so Sigmoid isn't trapped in the middle.
        #    (This is trainable, so the model can lower it if 20 is too high).
        # kernel_initializer=tf.keras.initializers.Constant(5.0),    
        #     bias_initializer=tf.keras.initializers.Constant(-2.2)
            )
        self.score_head  = keras.layers.Dense(
            1,
            activation=None,
            name="score",
            bias_initializer=tf.keras.initializers.Constant(0.049),
        )
        self.build_tower = build_tower # added new! 

    def encode_q(self, txt, did):
        return self.q_tower(
            self.concat([
                self.q_fs({"text": txt}),
                self.dise_emb(self.dise_lookup(did)),
            ])
        )

    def encode_k(self, txt, tid):
        txt_vec = self.k_fs({"text": txt})
        return self.k_tower(txt_vec)

    def call(self, feats):
        q = self.encode_q(
            feats["query"]["disease_text"],
            feats["query"]["diseaseId"],
        )
        k = self.encode_k(
            feats["candidate"]["target_text"],
            feats["candidate"]["targetId"],
        )
        sim  = self.dot([q, k])
        prob = self.cls_head(sim)
        reg  = self.score_head(sim)
        return {"cls": prob, "score": reg}

@keras.utils.register_keras_serializable() # added
def build_two_tower_model(df_learn) -> TwoTowerDual:
    # 1) Feature spaces
    q_fs = make_fs()
    k_fs = make_fs()

    q_fs.adapt(
        tf.data.Dataset.from_tensor_slices({"text": df_learn["disease_text"]})
          .batch(4096)
          .prefetch(tf.data.AUTOTUNE)
    )
    k_fs.adapt(
        tf.data.Dataset.from_tensor_slices({"text": df_learn["target_text"]})
          .batch(4096)
          .prefetch(tf.data.AUTOTUNE)
    )

    # 2) Lookup + embedding
    dise_lookup = keras.layers.StringLookup(name="disease_lookup")
    dise_lookup.adapt(df_learn["diseaseId"])
    dise_emb = keras.layers.Embedding(
        input_dim=dise_lookup.vocabulary_size(),
        output_dim=EMB_ID,
        name="dise_emb",
    )

    # # 3) Towers
    # # def build_tower(input_dim: int) -> keras.Model:
    # #     inp = keras.Input(shape=(input_dim + EMB_ID,))
    # #     # out = keras.layers.Dense(128)(inp)

    # #     out = keras.layers.Dense(128)(inp)
    # #     return keras.Model(inp, out, name="tower")
    # @keras.utils.register_keras_serializable() # added
    # def build_tower(input_dim: int,EMB_ID:int=64) -> keras.Model:
    #     inp = keras.Input(shape=(input_dim + EMB_ID,))
    #     x   = keras.layers.LayerNormalization()(inp)
    #     # x   = keras.layers.BatchNormalization()(inp)
    #     ## BatchNormalization
    #     # x = keras.layers.Dropout(0.1)(x)
    #     # x  = keras.layers.Dense(768, activation="gelu")(x)
    #     # out = keras.layers.Dense(256, activation="tanh")(x)
    #     # out = keras.layers.Dense(256, activation="gelu")(inp)
    #     out = keras.layers.Dense(256, activation="linear")(x)
    #     return keras.Model(inp, out, name="tower")

    q_tower = build_tower(q_fs.get_encoded_features().shape[-1])
    k_tower = build_tower(k_fs.get_encoded_features().shape[-1] - EMB_ID)

    concat = keras.layers.Concatenate(name="concat")

    # 4) Build model
    model = TwoTowerDual(
        dise_lookup=dise_lookup,
        dise_emb=dise_emb,
        q_fs=q_fs,
        k_fs=k_fs,
        q_tower=q_tower,
        k_tower=k_tower,
        concat_layer=concat,
        name="two_tower_dual",
    )

    # Dummy build
    dummy = {
        "query": {
            "disease_text": tf.constant(["dummy"]),
            "diseaseId": tf.constant([df_learn["diseaseId"].iloc[0]]),
        },
        "candidate": {
            "target_text": tf.constant(["dummy target"]),
            "targetId": tf.constant([df_learn["targetId"].iloc[0]]),
        },
    }
    _ = model(dummy)

    return model