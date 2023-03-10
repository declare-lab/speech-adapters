import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from text.symbols import symbols

from .constants import PAD
from .blocks import (
	get_sinusoid_encoding_table,
	LinearNorm,
)

import loralib as lora
from .adapters import PrefixEncoder, AdapterBlock, BottleneckAdapter

class TextEncoder(nn.Module):
	""" Text Encoder """

	def __init__(self, config):
		super(TextEncoder, self).__init__()

		n_position = config["max_seq_len"] + 1
		n_src_vocab = len(symbols) + 1
		d_word_vec = config["transformer"]["encoder_hidden"]
		n_layers = config["transformer"]["encoder_layer"]
		n_head = config["transformer"]["encoder_head"]
		d_k = d_v = (
			config["transformer"]["encoder_hidden"]
			// config["transformer"]["encoder_head"]
		)
		d_model = config["transformer"]["encoder_hidden"]
		d_inner = config["transformer"]["conv_filter_size"]
		kernel_size = config["transformer"]["conv_kernel_size"]
		dropout = config["transformer"]["encoder_dropout"]

		self.max_seq_len = config["max_seq_len"]
		self.d_model = d_model

		self.src_word_emb = nn.Embedding(
			n_src_vocab, d_word_vec, padding_idx=PAD
		)
		self.position_enc = nn.Parameter(
			get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
			requires_grad=False,
		)

		self.layer_stack = nn.ModuleList(
			[
				FFTBlock(
					config, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout, is_decoder=False
				)
				for _ in range(n_layers)
			]
		)

		self.config = config
		if config['adapter']['prefix_tuning']:
			self.num_heads = n_head
			self.prefix_seq_len = self.config["adapter"]["prefix_seq_len"]
			self.hidden_size = d_word_vec
			self.n_embd = self.hidden_size // self.num_heads
			self.prefix_tokens = torch.arange(self.prefix_seq_len).long()
			self.prefix_dropout = torch.nn.Dropout(config["adapter"]["prefix_dropout_prob"])
			self.num_layers = n_layers

			self.prefix_encoder = PrefixEncoder(config, num_hidden_layers=self.num_layers, hidden_size=self.hidden_size)

	def get_prefix_tuning(self, batch_size):
		
		prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)#.to(self.layer_norm.device)
		past_key_values = self.prefix_encoder(prefix_tokens)
		# bsz, seqlen, _ = past_key_values.shape
		past_key_values = past_key_values.view(
			batch_size,
			self.prefix_seq_len,
			self.num_layers * 2, 
			self.num_heads,
			self.n_embd
		)
		past_key_values = self.prefix_dropout(past_key_values)
		past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
		return past_key_values

	def forward(self, src_seq, mask, return_attns=False):  #(16, 231, 256)

		enc_slf_attn_list = []
		batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

		if self.config['adapter']['prefix_tuning']:
			past_key_values=self.get_prefix_tuning(batch_size=batch_size)
		else:
			past_key_values=None

		# -- Prepare masks
		slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

		# -- Forward
		src_word_emb = self.src_word_emb(src_seq)
		if not self.training and src_seq.shape[1] > self.max_seq_len:
			enc_output = src_word_emb + get_sinusoid_encoding_table(
				src_seq.shape[1], self.d_model
			)[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
				src_seq.device
			)
		else:
			enc_output = src_word_emb + self.position_enc[
				:, :max_len, :
			].expand(batch_size, -1, -1)

		for i, enc_layer in enumerate(self.layer_stack):

			past_key_value = past_key_values[i] if past_key_values is not None else None

			enc_output, enc_slf_attn = enc_layer(
				enc_output, mask=mask, slf_attn_mask=slf_attn_mask, past_key_value=past_key_value
			)
			if return_attns:
				enc_slf_attn_list += [enc_slf_attn]

		return enc_output, src_word_emb


class Decoder(nn.Module):
	""" Decoder """

	def __init__(self, config):
		super(Decoder, self).__init__()

		n_position = config["max_seq_len"] + 1
		d_word_vec = config["transformer"]["decoder_hidden"]
		n_layers = config["transformer"]["decoder_layer"]
		n_head = config["transformer"]["decoder_head"]
		d_k = d_v = (
			config["transformer"]["decoder_hidden"]
			// config["transformer"]["decoder_head"]
		)
		d_model = config["transformer"]["decoder_hidden"]
		d_inner = config["transformer"]["conv_filter_size"]
		kernel_size = config["transformer"]["conv_kernel_size"]
		dropout = config["transformer"]["decoder_dropout"]

		self.max_seq_len = config["max_seq_len"]
		self.d_model = d_model

		self.position_enc = nn.Parameter(
			get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
			requires_grad=False,
		)

		self.layer_stack = nn.ModuleList(
			[
				FFTBlock(
					config, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout, is_decoder=True
				)
				for _ in range(n_layers)
			]
		)

		self.config = config
		if config['adapter']['prefix_tuning']:
			self.num_heads = n_head
			self.prefix_seq_len = self.config["adapter"]["prefix_seq_len"]
			self.hidden_size = d_word_vec
			self.n_embd = self.hidden_size // self.num_heads
			self.prefix_tokens = torch.arange(self.prefix_seq_len).long()
			self.prefix_dropout = torch.nn.Dropout(config["adapter"]["prefix_dropout_prob"])
			self.num_layers = n_layers

			self.prefix_encoder = PrefixEncoder(config, num_hidden_layers=self.num_layers, hidden_size=self.hidden_size)

	def get_prefix_tuning(self, batch_size):
		
		prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)#.to(self.layer_norm.device)
		past_key_values = self.prefix_encoder(prefix_tokens)
		# bsz, seqlen, _ = past_key_values.shape
		past_key_values = past_key_values.view(
			batch_size,
			self.prefix_seq_len,
			self.num_layers * 2, 
			self.num_heads,
			self.n_embd
		)
		past_key_values = self.prefix_dropout(past_key_values)
		past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
		return past_key_values

	def forward(self, enc_seq, mask, return_attns=False):  # (16, 2842, 256)

		dec_slf_attn_list = []
		batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

		if self.config['adapter']['prefix_tuning']:
			past_key_values=self.get_prefix_tuning(batch_size=batch_size)
		else:
			past_key_values=None

		# -- Forward
		if not self.training and enc_seq.shape[1] > self.max_seq_len:
			# -- Prepare masks
			slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
			dec_output = enc_seq + get_sinusoid_encoding_table(
				enc_seq.shape[1], self.d_model
			)[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
				enc_seq.device
			)
		else:
			max_len = min(max_len, self.max_seq_len)

			# -- Prepare masks
			slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
			dec_output = enc_seq[:, :max_len, :] + self.position_enc[
				:, :max_len, :
			].expand(batch_size, -1, -1)
			mask = mask[:, :max_len]
			slf_attn_mask = slf_attn_mask[:, :, :max_len]

		for i, dec_layer in enumerate(self.layer_stack):

			past_key_value = past_key_values[i] if past_key_values is not None else None

			dec_output, dec_slf_attn = dec_layer(
				dec_output, mask=mask, slf_attn_mask=slf_attn_mask, past_key_value=past_key_value
			)
			if return_attns:
				dec_slf_attn_list += [dec_slf_attn]

		return dec_output, mask


class FFTBlock(nn.Module):
	""" FFT Block """

	def __init__(self, config, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1, is_decoder=False):
		super(FFTBlock, self).__init__()
		self.slf_attn = MultiHeadAttention(config, n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(
			d_model, d_inner, kernel_size, dropout=dropout
		)

		self.config = config
		if config["adapter"]["output_bottleneck"]:
			self.adapterblock = BottleneckAdapter("bottleneck_adapter", d_model, int(d_model/2))
		elif config["adapter"]["conv_adapter"]:
			self.adapterblock = AdapterBlock(d_model, int(d_model/2))

	def forward(self, enc_input, mask=None, slf_attn_mask=None, past_key_value=None):

		enc_output, enc_slf_attn = self.slf_attn(
			enc_input, enc_input, enc_input, mask=slf_attn_mask, past_key_value=past_key_value
		)
		if mask is not None:
			enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

		enc_output = self.pos_ffn(enc_output)

		if self.config["adapter"]["output_bottleneck"]:
			enc_output = self.adapterblock(x=enc_output, residual_input=enc_output)
		elif self.config["adapter"]["conv_adapter"]:
			enc_output = self.adapterblock(x=enc_output, residual_input=enc_output)
		elif self.config["adapter"]["tiny_attention"]:
			enc_output = enc_output + self.tiny_attn(hidden_states=enc_output)
		elif self.config["adapter"]["tiny_external_attention"]:
			enc_output = enc_output + self.tiny_attn(hidden_states=enc_output)
		elif self.config["adapter"]["tiny_conformer"]:
			enc_output = enc_output + self.tiny_conformer(hidden_states=enc_output)

		if mask is not None:
			enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

		return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):
	""" Multi-Head Attention """

	def __init__(self, config, n_head, d_model, d_k, d_v, dropout=0.1):
		super(MultiHeadAttention, self).__init__()

		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v

		self.config = config

		if config["adapter"]["lora"]:
			self.w_qs = lora.Linear(d_model, n_head * d_k, r=8)
			self.w_ks = LinearNorm(d_model, n_head * d_k)
			self.w_vs = lora.Linear(d_model, n_head * d_v, r=8)
		else:
			self.w_qs = LinearNorm(d_model, n_head * d_k)
			self.w_ks = LinearNorm(d_model, n_head * d_k)
			self.w_vs = LinearNorm(d_model, n_head * d_v)

		self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
		self.layer_norm = nn.LayerNorm(d_model)

		self.fc = LinearNorm(n_head * d_v, d_model)

		self.dropout = nn.Dropout(dropout)

	def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int, dim: int):
		return tensor.view(bsz, seq_len, self.n_head, dim).transpose(1, 2).contiguous()

	def forward(self, q, k, v, mask=None, past_key_value=None):

		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

		sz_b, len_q, _ = q.size()
		sz_b, len_k, _ = k.size()
		sz_b, len_v, _ = v.size()

		residual = q

		batch_size = q.size(0)
		
		if past_key_value is not None:    # prefix-tuning
			mask = mask.repeat(n_head, 1, 1)

			key_states = self._shape(self.w_ks(k), -1, batch_size, self.d_k)
			value_states = self._shape(self.w_vs(v), -1, batch_size, self.d_v)
			key_states = torch.cat([past_key_value[0], key_states], dim=2)
			value_states = torch.cat([past_key_value[1], value_states], dim=2)

			prefix_attention_mask = torch.ones(batch_size, self.config['adapter']['prefix_seq_len']).to(mask.device)
			prefix_attention_mask = 1.0 - prefix_attention_mask
			prefix_attention_mask = prefix_attention_mask[:, None, :].repeat(n_head, mask.size(-1), 1)
			
			mask = torch.tensor(torch.cat((prefix_attention_mask, mask), dim=-1), dtype=torch.bool)
			
			q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
			len_k = key_states.size(2)
			len_v = value_states.size(2)
			k = key_states.permute(1,0,2,3).contiguous().view(-1, len_k, d_k)
			v = value_states.permute(1,0,2,3).contiguous().view(-1, len_v, d_v)
		
			output, attn = self.attention(q, k, v, mask=mask)
		
		else:

			q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
			k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
			v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
			q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
			k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
			v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

			mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
			output, attn = self.attention(q, k, v, mask=mask)

		output = output.view(n_head, sz_b, len_q, d_v)
		output = (
			output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
		)  # b x lq x (n*dv)

		output = self.dropout(self.fc(output))
		output = self.layer_norm(output + residual)

		return output, attn


class ScaledDotProductAttention(nn.Module):
	""" Scaled Dot-Product Attention """

	def __init__(self, temperature):
		super(ScaledDotProductAttention, self).__init__()
		self.temperature = temperature
		self.softmax = nn.Softmax(dim=2)

	def forward(self, q, k, v, mask=None):

		attn = torch.bmm(q, k.transpose(1, 2))
		attn = attn / self.temperature

		if mask is not None:
			attn = attn.masked_fill(mask, -np.inf)

		attn = self.softmax(attn)
		output = torch.bmm(attn, v)

		return output, attn


class PositionwiseFeedForward(nn.Module):
	""" A two-feed-forward-layer """

	def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()

		# Use Conv1D
		# position-wise
		self.w_1 = nn.Conv1d(
			d_in,
			d_hid,
			kernel_size=kernel_size[0],
			padding=(kernel_size[0] - 1) // 2,
		)
		# position-wise
		self.w_2 = nn.Conv1d(
			d_hid,
			d_in,
			kernel_size=kernel_size[1],
			padding=(kernel_size[1] - 1) // 2,
		)

		self.layer_norm = nn.LayerNorm(d_in)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		residual = x
		output = x.transpose(1, 2)
		output = self.w_2(F.relu(self.w_1(output)))
		output = output.transpose(1, 2)
		output = self.dropout(output)
		output = self.layer_norm(output + residual)

		return output
