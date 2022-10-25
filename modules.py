import torch
from torch import nn
from typing import *

from transformers import Trainer
from transformers.adapters import AdapterConfig, CompacterConfig, PrefixTuningConfig
from transformers.adapters.modeling import Adapter
from transformers.adapters.prefix_tuning import PrefixTuningShim, PrefixTuningPool#, PrefixTuning

from transformers.adapters.wrappers.configuration import wrap_config

from transformers.activations import ACT2FN

class SELayer4Vision(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y.expand_as(x)


class SELayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool1d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		resdiual = x
		b, c, _= x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1)
		# return resdiual + x * y.expand_as(x)
		return x * y.expand_as(x)
'''
def depthwise_conv5X5(in_planes, out_planes, stride=1):
	return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride, bias=False, groups=4)

def conv1x1(in_planes, out_planes, stride=1):
	"1x1 convolution without padding"
	return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
'''

class PrefixEncoder(nn.Module):  
	#code from P-tuning-v2
	#https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.prefix_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.prefix_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class ConvAdapter(nn.Module):
	def __init__(self, config, in_channels, out_channels, kernel_size=1, stride=1, bias=False, layer_id=0):
		super(ConvAdapter, self).__init__()
		self.conv_down = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)#, padding='same')
		# self.layer_norm_down = nn.LayerNorm(config.hidden_size, elementwise_affine=True) # size will change in each feature encoder(conv layer)
		self.activation_down = ACT2FN[config.feat_extract_activation]
		if layer_id == 0:
			self.conv_up = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, bias=bias, padding='same')
		else:
			self.conv_up = nn.Conv1d(out_channels, in_channels, kernel_size=1, stride=1, bias=bias, padding='same')
		# self.conv_up = nn.Conv1d(out_channels, in_channels, kernel_size=1, stride=1, bias=bias, padding='same')
		# self.layer_norm_up = nn.LayerNorm(config.hidden_size, elementwise_affine=True)   # need to confirm
		# self.activation_up = ACT2FN[config.feat_extract_activation]
	def forward(self, x, residual_input):
		out = self.conv_down(x)
		# out = self.layer_norm_down(out)
		out = self.activation_down(out)
		out = self.conv_up(out)
		# out = out*0 + residual_input
		out = out + residual_input
		return out

class AdapterBlock(nn.Module):
	def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, bias=False):
		super(AdapterBlock, self).__init__()
		self.layer_norm1 = nn.LayerNorm(1024)
		self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=stride, bias=bias, padding='same')
		self.relu1 = nn.ReLU(inplace=True)
		# self.se1 = SELayer(out_dim)
		self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=5, stride=stride, bias=False, groups=out_dim, padding='same')
		# self.se2 = SELayer(out_dim)
		self.conv3 = nn.Conv1d(out_dim, in_dim, kernel_size=3, stride=stride, bias=bias, padding='same')
		# self.relu2 = nn.ReLU(inplace=True)
		self.se3 = SELayer(out_dim)
		self.layer_norm2 = nn.LayerNorm(out_dim)
	def forward(self, x, residual_input):
		out = self.layer_norm1(x)
		out = self.conv1(out)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.se3(out)
		out = residual_input + out   #skip connection
		return out


# class AdapterBlock(nn.Module):
# 	def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, bias=False):
# 		super(AdapterBlock, self).__init__()
# 		self.layer_norm1 = nn.LayerNorm(out_dim)
# 		self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=stride, bias=bias, padding='same')
# 		self.relu1 = nn.ReLU(inplace=True)
# 		# self.se1 = SELayer(out_dim)
# 		self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=5, stride=stride, bias=False, groups=124, padding='same')
# 		# self.se2 = SELayer(out_dim)
# 		self.conv3 = nn.Conv1d(out_dim, in_dim, kernel_size=3, stride=stride, bias=bias, padding='same')
# 		# self.relu2 = nn.ReLU(inplace=True)
# 		self.se3 = SELayer(out_dim)
# 		self.layer_norm2 = nn.LayerNorm(out_dim)
# 	def forward(self, x, residual_input):
# 		x = x.transpose(-2, -1)
# 		out = self.layer_norm1(x)
# 		out = out.transpose(-2, -1)
# 		out = self.conv1(out)
# 		out = self.relu1(out)
# 		# out = self.se1(out)
# 		out = self.conv2(out)
# 		# out = self.se2(out)
# 		out = self.conv3(out)
# 		# out = self.relu2(out)
# 		out = self.se3(out)
# 		out = residual_input + out   #skip connection
# 		out = out.transpose(-2, -1)
# 		out = self.layer_norm2(out)
# 		out = out.transpose(-2, -1)
# 		return out

class BottleneckAdapter(nn.Module):
	def __init__(self, adapter_name, input_size, down_sample):
		super(BottleneckAdapter, self).__init__()
		self.config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
		self.bottleneck_adapter = Adapter(adapter_name, input_size=input_size, down_sample=down_sample, config=self.config)
	def forward(self, x, residual_input):
		output, down, up = self.bottleneck_adapter(x, residual_input)
		return output

''' didn't use
class PrefixTuningAdapter(nn.Module):
	def __init__(self, config, n_layers, n_heads, input_size, location_key=None):
		super(PrefixTuningAdapter, self).__init__()
		prefix_config = PrefixTuningConfig(flat=False, prefix_length=30)
		config.model_type = "wav2vec2"
		self.config = wrap_config(config)
		# self.prefixtuning = PrefixTuning(n_layers, n_heads, input_size, self.config)
		# print(self.config.adapters.__dict__)
		if "prefix_tuning" not in self.config.adapters:
			self.config.adapters.add("prefix_tuning", config=prefix_config)
		# self.config.adapters["prefix_tuning"] = "648bf22f5afeaaa6"   ## {adapter_name:config_name}
		pool = PrefixTuningPool(self.config)
		# print("Successful!!!")
		self.prefix_tuning = PrefixTuningShim(location_key + "_prefix" if location_key else None, self.config)
		self.prefix_tuning.set_pool(pool)
		# self.prefix_tuning.pool.confirm_prefix("prefix_tuning")
	def forward(self, keys, values, hidden_states, attention_mask):
		keys, values, attention_mask=self.prefix_tuning(keys, values, hidden_states, attention_mask)
		return keys, values, attention_mask


class CompacterAdapter(nn.Module):
	def __init__(self, adapter_name, input_size, down_sample):
		super(CompacterAdapter, self).__init__()
		self.config = CompacterConfig()
		self.compacter_adapter = Adapter(adapter_name, input_size=input_size, down_sample=down_sample, config=self.config)
	def forward(self, x, residual_input):
		print("=============CompacterAdapter==========")
		print("x:", x.size())
		print("residual_input:", residual_input.size())
		output, down, up = self.compacter_adapter(x, residual_input)
		print("output: ", output.size())
		print("down  : ", down.size())
		return output
'''


class CustomTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs.get("labels")
		# forward pass
		outputs = model(**inputs)
		logits = outputs.get("logits")

		# compute custom loss (suppose one has 3 labels with different weights)
		# loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0])).to(labels.device)  #add weight or not?
		loss_fct = nn.CrossEntropyLoss().to(labels.device)
		loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
		return (loss, outputs) if return_outputs else loss

'''
if __name__ == "__main__":
	model = PrefixTuningAdapter(13,13,1024)
	print(model)
'''

