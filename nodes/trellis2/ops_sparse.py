"""
Thin re-export of comfy.ops_sparse (= comfy_sparse_attn package, junctioned at load time).
"""
from comfy.ops_sparse import (  # noqa: F401
    disable_weight_init,
    manual_cast,
    get_conv_backend,
    set_conv_backend,
)
