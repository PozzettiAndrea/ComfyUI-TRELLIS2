"""
Thin re-export of comfy.attention_sparse (= comfy_sparse_attn package, junctioned at load time).
"""
from comfy.attention_sparse import (  # noqa: F401
    scaled_dot_product_attention,
    sparse_scaled_dot_product_attention,
    dispatch_varlen_attention,
)
