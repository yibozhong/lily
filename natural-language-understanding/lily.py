import torch
from torch import nn
import inspect
import math
from typing import Optional, Tuple
import torch.nn.functional as F

class lily_adapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, ne, lp, hps, mlp=False, idx=4):
        super().__init__()
        self.hps = hps
        self.ne = ne
        self.lp = lp
        self.router = nn.Linear(hidden_dim, ne, bias=False)
        if mlp:
            self.non_linear = nn.ReLU()
        else:
            self.non_linear = nn.Identity()
        self.idx = idx
    def forward(self, x):
        hidden = self.non_linear(self.lp(x))
        router_logits = self.router(hidden) # [B, N, num_of_experts]
        router_probability = F.softmax(router_logits, dim=-1) # [B, N, ne]
        expert_probabilities = router_probability.mean(dim=(0, 1)) 
        combined_hp = torch.einsum("e,eio->io", expert_probabilities, self.hps)
        return torch.matmul(hidden, combined_hp)
    
class lily_adapter_monoscale(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, ne, lp, hps, mlp=False):
        super().__init__()
        self.hps = hps
        self.ne = ne
        self.lp = lp
        self.scale = 1 / ne
        if mlp:
            self.non_linear = nn.ReLU()
        else:
            self.non_linear = nn.Identity()
    def forward(self, x):
        hidden = self.non_linear(self.lp(x))
        combined_hp = torch.sum(self.hps, 0) * self.scale
        return torch.matmul(hidden, combined_hp)

def feed_forward_chunk_lily(self, attention_output):
        delta_mlp = self.lily_mlp(attention_output)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output + delta_mlp
    
def forward_attn_lily(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states) + (self.lily_q(hidden_states) * self.s)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states) + (self.lily_v(encoder_hidden_states) * self.s))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states) + (self.lily_v(hidden_states) * self.s))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states) + (self.lily_v(hidden_states) * self.s))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

def set_lily(model, dim=8, ne=4, s=1.0):
    lily_q_hps = nn.Parameter(torch.zeros(ne, dim, 768))
    lily_v_hps = nn.Parameter(torch.zeros(ne, dim, 768))
    previous_q_lp = None
    previous_v_lp = None
    idx = 0
    stride = 12 // ne
    for n, layer in model.named_modules():
        if n.endswith(('self')):
            if idx % stride == 0:
                print(f"set new lp at layer {idx}")
                previous_q_lp = nn.Linear(768, dim, bias=False)
                previous_v_lp = nn.Linear(768, dim, bias=False)
            layer.lily_q = lily_adapter(768, 768, dim, ne, previous_q_lp, lily_q_hps)
            layer.lily_v = lily_adapter(768, 768, dim, ne, previous_v_lp, lily_v_hps)
            layer.s = s
            bound_method = forward_attn_lily.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
            idx += 1
        