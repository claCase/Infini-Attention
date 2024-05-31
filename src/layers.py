from packaging.version import Version
import tensorflow as tf
from keras import layers, models, activations
if Version(tf.__version__) >= Version("2.16.0"):
    from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell as DropCell
    from keras_nlp.api.layers import SinePositionEncoding
else:
    from keras_nlp.layers import SinePositionEncoding
    from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin as DropCell
import numpy as np

class AttentionRNNCell(
    tf.keras.layers.Layer,
    DropCell, 
):
    def __init__(
        self,
        dims,
        heads,
        context_length,
        delta_rule=True,
        dropout=0,
        initializer="glorot_uniform",
        causal=True,
        positional_encoding=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dims = dims
        self.heads = heads
        self.context_length = context_length
        self.state_size = [
            tf.TensorShape((heads, dims, dims)),  # Memory
            tf.TensorShape((heads, dims)),  # Normalization
        ]
        self.output_size = tf.TensorShape((context_length, dims))
        self.delta_rule = delta_rule
        self.dropout = dropout
        self.causal = causal 
        if positional_encoding:
            self.positional = SinePositionEncoding(max_wavelength=1000, dtype=tf.float32)
        else:
            self.positional = None 
    def build(self, input_shape):
        i = input_shape[-1]
        self.attn_kernel = self.add_weight(
            name="kvq_kernel", shape=(i, self.heads, self.dims, 3)
        )
        self.out_kernel = self.add_weight(
            name="kvq_kernel", shape=(self.dims, self.heads, self.dims)
        )
        self.beta = self.add_weight(name="beta", shape=(1,))

    def call(self, inputs, states, training):
        """Call method of layer

        Args:
            inputs (tf.Tensor): (B, N, D) where B is the batch size, N is the context length, D is the input dimension
            states (Tuple[tf.Tensor, tf.Tensor]): [(B, H, O, O), (B, H, O)] where H is the number of heads, O is the output dimension
            training (bool): Flag True if training

        Returns:
            _type_: _description_
        """
        if self.positional is not None:
            inputs = inputs + tf.cast(self.positional(inputs), inputs.dtype)
        mem, z = states
        kvq = tf.einsum("bni,ihok->bnhok", inputs, self.attn_kernel)
        if self.dropout > 0:
            kvq_drop = self.get_dropout_mask_for_cell(
                inputs=kvq, training=True, count=1
            )
            kvq = kvq * kvq_drop

        k, v, q = tf.split(kvq, 3, -1)
        k, v, q = k[..., 0], v[..., 0], q[..., 0]

        # Retrive context from memory
        q_elu = tf.nn.elu(q) + 1.0
        A_mem_0 = tf.einsum("bhdo,bnhd->bnho", mem, q_elu)
        A_mem_1 = tf.einsum("bhd,bnhd->bnh", z, q_elu)
        A_mem = A_mem_0 / (A_mem_1[..., None] + 1e-8)

        # Update Memory
        k_elu = tf.nn.elu(k) + 1.0
        if self.delta_rule:
            next_mem = mem + tf.einsum("bnhz,bnho->bhzo", k_elu, v - A_mem)
        else:
            next_mem = mem + tf.einsum("bnhz,bnho->bhzo", k_elu, v)

        # Update normalizer
        next_z = z + tf.reduce_sum(k_elu, 1)

        # Compute standard attention
        qk = tf.einsum("bnho,bkho->bhnk", q, k)
        d = tf.math.sqrt(tf.cast(self.attn_kernel.shape[0], inputs.dtype))
        qk_normed = qk / d
        if self.causal:
            mask = tf.ones_like(qk_normed)
            mask = -(1. - tf.linalg.LinearOperatorLowerTriangular(mask).to_dense())*1e10
            qk_normed = qk_normed + mask 
        A_soft = tf.nn.softmax(qk_normed)
        A_dot = tf.einsum("bhnk,bkho->bnho", A_soft, v)

        # Long term Context Injection
        A = tf.nn.sigmoid(self.beta) * A_mem + (1 - tf.nn.sigmoid(self.beta)) * A_dot

        # Output
        O = tf.einsum("bnhi,iho", A, self.out_kernel)
        return O, [next_mem, next_z]
