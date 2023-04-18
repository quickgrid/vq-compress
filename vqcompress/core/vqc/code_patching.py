from einops import rearrange

from vqcompress.core import shared

try:
    # import bitsandbytes as bnb
    # shared.Config.available_bitsandbytes = True
    import xformers.ops
    shared.Config.available_xformers = True
except ModuleNotFoundError as err:
    print(err)


def get_xformers_flash_attention_op(q, k, v):
    try:
        flash_attention_op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
        # flash_attention_op = xformers.ops.MemoryEfficientAttentionOp
        fw, bw = flash_attention_op
        if fw.supports(xformers.ops.fmha.Inputs(query=q, key=k, value=v, attn_bias=None)):
            return flash_attention_op
    except Exception as e:
        print(e, "enabling flash attention")

    return None


def patch_xformers_attn_forward(self, x):
    """Can replace LDM vqgan attention method forward with xformers attention. Should also work when replacing other
    attention code. Code copied from,
    https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_optimizations.py.

    Examples:
        >>> import vqcompress.core.vqc.code_patching
        >>> import vqcompress.core.ldm.model
        >>> vqcompress.core.ldm.model.AttnBlock.forward = vqcompress.core.vqc.code_patching.patch_xformers_attn_forward
    """
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
    # dtype = q.dtype
    # if True:
    #     q, k = q.float(), k.float()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    # out = xformers.ops.memory_efficient_attention(q, k, v, op=get_xformers_flash_attention_op(q, k, v))
    out = xformers.ops.memory_efficient_attention(q, k, v)
    # out = out.to(dtype)
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    out = self.proj_out(out)
    return x + out
