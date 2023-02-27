import torch
from torch import nn
from typing import *

from transformers import Trainer
from transformers.adapters import AdapterConfig
from transformers.adapters.modeling import Adapter

from transformers.activations import ACT2FN

from torch.nn import init
from torch import Tensor

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

class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        return x

class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs):
        return inputs * inputs.sigmoid()

class FeedForwardModule(nn.Module):
	"""
	Conformer Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
	and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
	regularizing the network.
	Args:
		encoder_dim (int): Dimension of conformer encoder
		expansion_factor (int): Expansion factor of feed forward module.
		dropout_p (float): Ratio of dropout
	Inputs: inputs
		- **inputs** (batch, time, dim): Tensor contains input sequences
	Outputs: outputs
		- **outputs** (batch, time, dim): Tensor produces by feed forward module.
	"""
	def __init__(
			self,
			encoder_dim: int = 512,
			expansion_factor: float = 4,
			dropout_p: float = 0.1,
	) -> None:
		super(FeedForwardModule, self).__init__()
		self.sequential = nn.Sequential(
			nn.LayerNorm(encoder_dim),
			# LinearNorm(encoder_dim, encoder_dim, bias=True),
			LinearNorm(encoder_dim, int(encoder_dim * expansion_factor), bias=True),
			Swish(),
			nn.Dropout(p=dropout_p),
			# LinearNorm(int(encoder_dim * expansion_factor), encoder_dim, bias=True),
			# nn.Dropout(p=dropout_p),
		)

	def forward(self, inputs: Tensor, past_key_value=None) -> Tensor:
		return self.sequential(inputs)

class AdapterBlock(nn.Module):
	def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, bias=False):
		super(AdapterBlock, self).__init__()
		self.layer_norm1 = nn.LayerNorm(in_dim)
		self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=stride, bias=bias,groups=out_dim, padding='same')
		self.relu1 = nn.ReLU(inplace=True)
		# self.se1 = SELayer(out_dim)
		self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=5, stride=stride, bias=False, groups=out_dim, padding='same')
		# self.se2 = SELayer(out_dim)
		self.conv3 = nn.Conv1d(out_dim, in_dim, kernel_size=3, stride=stride, bias=bias,groups=out_dim, padding='same')
		# self.relu2 = nn.ReLU(inplace=True)
		self.se3 = SELayer(in_dim)
		# self.layer_norm2 = nn.LayerNorm(out_dim)
		# self.dropout = nn.Dropout(p=0.1)
	def forward(self, x, residual_input):
		out = self.layer_norm1(x)
		out = torch.transpose(out,-1,-2)
		out = self.conv1(out)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.se3(out)
		# out = self.dropout(out)
		out = torch.transpose(out,-1,-2)
		out = residual_input + out   #skip connection
		return out

class BottleneckAdapter(nn.Module):
	def __init__(self, adapter_name, input_size, down_sample):
		super(BottleneckAdapter, self).__init__()
		self.config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
		self.bottleneck_adapter = Adapter(adapter_name, input_size=input_size, down_sample=down_sample, config=self.config)
	def forward(self, x, residual_input):
		output, down, up = self.bottleneck_adapter(x, residual_input)
		return output


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



