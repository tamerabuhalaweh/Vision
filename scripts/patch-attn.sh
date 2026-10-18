#!/bin/bash
set -u
cd /home/administrator/vision-work/Wan2.2
[ -f wan/modules/attention.py.orig ] || cp wan/modules/attention.py wan/modules/attention.py.orig
/home/administrator/vision-work/venv-wan/bin/python - <<'PY'
import pathlib
p = pathlib.Path("wan/modules/attention.py")
src = p.read_text()
old = """    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func("""
new = """    else:
        if not FLASH_ATTN_2_AVAILABLE:
            warnings.warn('flash-attn unavailable -> SDPA fallback in flash_attention()')
            q3 = q.unflatten(0, (b, lq)).transpose(1, 2).to(dtype)
            k3 = k.unflatten(0, (b, lk)).transpose(1, 2).to(dtype)
            v3 = v.unflatten(0, (b, lk)).transpose(1, 2).to(dtype)
            out = torch.nn.functional.scaled_dot_product_attention(
                q3, k3, v3, attn_mask=None, is_causal=causal, dropout_p=dropout_p,
                scale=softmax_scale)
            return out.transpose(1, 2).contiguous().type(out_dtype)
        x = flash_attn.flash_attn_varlen_func("""
assert old in src, "anchor not found"
p.write_text(src.replace(old, new, 1))
print("patched OK")
PY
