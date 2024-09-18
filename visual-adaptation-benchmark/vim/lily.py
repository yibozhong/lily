import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from einops import rearrange, repeat
from causal_conv1d import causal_conv1d_fn
import causal_conv1d_cuda
import selective_scan_cuda

class lily_hp(nn.Module):
    def __init__(self, num_of_experts, in_dim, out_dim=4):
        super().__init__()
        # router
        self.router = nn.Linear(in_dim, num_of_experts, bias=False)
        # all HP experts
        self.adapters = nn.Parameter(torch.zeros(num_of_experts, in_dim, out_dim))
        
    def forward(self, tokens):
        # get probabilities for all experts and combine them into a single adapter
        # tokens [B, L, D]
        router_logits = self.router(tokens) # [B, L, num_of_experts]
        router_probability = F.softmax(router_logits, dim=-1) # [B, L, num_of_experts]
        combined_adapter = torch.einsum("ble,eio->blio", router_probability, self.adapters)
        return combined_adapter
     
class MambaInnerFnNoOutProjAdapter(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                    C_proj_bias=None, delta_softplus=True,
                    adapt_delta_proj=False, adapter_delta_proj_weight=None, checkpoint_lvl=1, s=1):
            
            assert checkpoint_lvl in [0, 1]
            L = xz.shape[-1]
            delta_rank = delta_proj_weight.shape[1]
            d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
            if torch.is_autocast_enabled():
                x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
                delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
                # in case using adapters
                if adapt_delta_proj:
                    adapter_delta_proj_weight = adapter_delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            if xz.stride(-1) != 1:
                xz = xz.contiguous()
            conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
            x, z = xz.chunk(2, dim=1)
            conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias,None, True)
            x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
            if adapt_delta_proj:
                delta += rearrange(adapter_delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L) 
            
            ctx.is_variable_B = B is None
            ctx.is_variable_C = C is None
            ctx.B_proj_bias_is_None = B_proj_bias is None
            ctx.C_proj_bias_is_None = C_proj_bias is None
            if B is None:  # variable B
                B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
                if B_proj_bias is not None:
                    B = B + B_proj_bias.to(dtype=B.dtype)
                if not A.is_complex():
                    # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                    B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
                else:
                    B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
            else:
                if B.stride(-1) != 1:
                    B = B.contiguous()
            if C is None:  # variable C
                C = x_dbl[:, -d_state:]  # (bl dstate)
                if C_proj_bias is not None:
                    C = C + C_proj_bias.to(dtype=C.dtype)
                if not A.is_complex():
                    # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                    C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
                else:
                    C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
            else:
                if C.stride(-1) != 1:
                    C = C.contiguous()
            if D is not None:
                D = D.contiguous()
            out, scan_intermediates, out_z = selective_scan_cuda.fwd(
                conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
            )
            ctx.delta_softplus = delta_softplus
            ctx.checkpoint_lvl = checkpoint_lvl
            if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
                conv1d_out, delta = None, None
            # return rearrange(out_z, "b d l -> b l d")
            ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, out, adapter_delta_proj_weight)
            return out_z
    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, adapter_delta_proj_weight) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias,None, True)
            # recompute adaptation results
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                                "d (b l) -> b d l", l = L)
            delta += rearrange(adapter_delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
            
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        # dout_y = rearrange(dout, "b l d -> b d l") # because no arrange at end of forward, so dout shape is b d l
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, dx, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                # dB_proj_bias, dC_proj_bias, None, None, ddelta_proj_weight, None, None, None)
                dB_proj_bias, dC_proj_bias, None, None, ddelta_proj_weight, None, None) # no need to return ddelta_proj_weight

def mamba_inner_fn_no_out_proj_adapter(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True,adapt_delta_proj=False, adapter_delta_proj_weight=None, checkpoint_lvl=1, s=1):
    return MambaInnerFnNoOutProjAdapter.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, adapt_delta_proj, adapter_delta_proj_weight, checkpoint_lvl, s)
def forward_mamba(self, hidden_states, inference_params=None):
    """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
    """
    batch, seqlen, dim = hidden_states.shape
    # print(self.delta_adapter_down.weight.grad)
    conv_state, ssm_state = None, None
    if inference_params is not None:
        conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
        if inference_params.seqlen_offset > 0:
            # The states are updated inplace
            out, _, _ = self.step(hidden_states, conv_state, ssm_state)
            return out
    # We do matmul and transpose BLH -> HBL at the same time
    xz = rearrange(
        self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        "d (b l) -> b d l",
        l=seqlen,
    )
    if self.adapt_in:
        hidden_states_after_adapter_down = self.in_adapter_down(hidden_states) # [B, L, hidden_dim]
        if self.using_dropout:
            hidden_states_after_adapter_down = self.dropout(hidden_states_after_adapter_down)
        combined_adapter_up = self.in_adapter_up(hidden_states_after_adapter_down) # [B, L, hidden_dim, feature_dim]
        hidden_states_after_adapter_up = torch.einsum("bld,blde->ble", hidden_states_after_adapter_down, combined_adapter_up) # [B, L, D]
        xz += rearrange(hidden_states_after_adapter_up, "b l d -> b d l") # back to its originial shape

    if self.in_proj.bias is not None:
        xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
    # In the backward pass we write dx and dz next to each other to avoid torch.cat
    if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
        A_b = -torch.exp(self.A_b_log.float())
        out = mamba_inner_fn_no_out_proj_adapter(
            xz,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            A,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            adapt_delta_proj=self.adapt_delta,
            adapter_delta_proj_weight=(self.delta_adapter_down.weight.t() @ (self.delta_adapter_up.weight.t())).t(),
            s=self.s,
        )
        out_b = mamba_inner_fn_no_out_proj_adapter(
            xz.flip([-1]),
            self.conv1d_b.weight,
            self.conv1d_b.bias,
            self.x_proj_b.weight,
            self.dt_proj_b.weight,
            A_b,
            None,
            None,
            self.D_b.float(),
            delta_bias=self.dt_proj_b.bias.float(),
            delta_softplus=True,
            adapt_delta_proj=self.adapt_delta,
            adapter_delta_proj_weight=(self.delta_b_adapter_down.weight.t() @ (self.delta_b_adapter_up.weight.t())).t(),
            s=self.s,
        )
        # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
        origin_out = out
        if not self.if_devide_out:
            out = F.linear(rearrange(origin_out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
        else:
            out = F.linear(rearrange(origin_out + out_b.flip([-1]), "b d l -> b l d") / 2, self.out_proj.weight, self.out_proj.bias)

    if self.init_layer_scale is not None:
            out = out * self.gamma    

    return out


def set_lily(model, in_dim=8, delta_dim=8, adapt_delta=False, adapt_in=False, s=1, ne=1, dropout=False):
    if adapt_delta:
        # hps for delta 
        model.shared_delta_adapter_up = nn.Linear(delta_dim, 768, bias=False)
        model.shared_delta_b_adapter_up = nn.Linear(delta_dim, 768, bias=False)
        nn.init.zeros_(model.shared_delta_adapter_up.weight)
        nn.init.zeros_(model.shared_delta_b_adapter_up.weight)
    
    if adapt_in:
        model.shared_in_adapter_up = lily_hp(ne, in_dim, 1536)
    
    # setting the projectors, note that LPs are not shared like in Transformer.
    for name, layer in model.named_modules():
        if 'mixer' in name and 'in_proj' not in name and 'conv1d' not in name and 'act' not in name and 'x_proj_b' not in name and 'x_proj' not in name and 'dt_proj' not in name and 'dt_proj_b' not in name and 'conv1d_b' not in name and 'out_proj' not in name and 'adapter' not in name and 'dropout' not in name:
            # print(name)
            if dropout:
                layer.dropout = nn.Dropout(0.1)
            else:
                layer.dropout = nn.Dropout(0.0)
            layer.adapt_delta = adapt_delta
            layer.adapt_in = adapt_in
            layer.s = s # not used here. In other words, s is always 1 in Vim VTAB-1K
            layer.using_dropout = dropout
            if adapt_delta:
                layer.delta_adapter_down = nn.Linear(layer.dt_rank, delta_dim, bias=False)
                layer.delta_b_adapter_down = nn.Linear(layer.dt_rank, delta_dim, bias=False)
                layer.delta_adapter_up = model.shared_delta_adapter_up 
                layer.delta_b_adapter_up = model.shared_delta_b_adapter_up
            if adapt_in:
                layer.in_adapter_down = nn.Linear(layer.d_model, in_dim, bias=False)
                layer.in_adapter_up = model.shared_in_adapter_up
            bound_method = forward_mamba.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)