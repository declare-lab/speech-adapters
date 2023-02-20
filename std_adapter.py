import torch
from transformers import (Wav2Vec2ForSequenceClassification, 
							Wav2Vec2Config, 
							Wav2Vec2Model, 
							Wav2Vec2Processor,
							Wav2Vec2PreTrainedModel)
from transformers.adapters import AdapterConfig
from transformers.adapters.model_mixin  import EmbeddingAdaptersWrapperMixin
from transformers.adapters.heads import (ModelWithFlexibleHeadsAdaptersMixin,
										 MultiLabelClassificationHead,
										  ClassificationHead)

from transformers.models.wav2vec2.modeling_wav2vec2 import WAV_2_VEC_2_INPUTS_DOCSTRING, WAV_2_VEC_2_START_DOCSTRING
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward

import utils
from data import EsdDataset
from torch.utils.data import Dataset, DataLoader

from transformers.adapters.context import AdapterSetup
from adapter.Adapter import Adapter

@add_start_docstrings(
	"The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.",
	WAV_2_VEC_2_START_DOCSTRING,
)
class Wav2Vec2AdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, Wav2Vec2PreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.wav2vec2 = Wav2Vec2Model(config)

		self._init_head_modules()

		self.init_weights()
	
	@add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
	def forward(
		self,
		input_values = None,
		attention_mask = None,
		mask_time_indices = None,
		output_attentions = None,
		output_hidden_states = None,
		return_dict = None,
		head=None,
		output_adapter_gating_scores=False,
		output_adapter_fusion_attentions=False,
		**kwargs
	):
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		outputs = self.wav2vec2(
			input_values,
			attention_mask=attention_mask,
			mask_time_indices=mask_time_indices,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			output_adapter_gating_scores=output_adapter_gating_scores,
			output_adapter_fusion_attentions=output_adapter_fusion_attentions
		)

		if not return_dict:
			head_inputs = (outputs[0],)
		else:
			head_inputs = outputs

		if head or AdapterSetup.get_context_head_setup() or self.active_head:
			head_outputs = self.forward_head(
				head_inputs,
				head_name=head,
				attention_mask=attention_mask,
				return_dict=return_dict,
				pooled_output=pooled_output,
				**kwargs,
			)
			return head_outputs
		else:
			# in case no head is used just return the output of the base model (including pooler output)
			return outputs

	head_types = {
		"classification": ClassificationHead,
		"multilabel_classification": MultiLabelClassificationHead,
		# "tagging": TaggingHead,
		# "multiple_choice": MultipleChoiceHead,
		# "question_answering": QuestionAnsweringHead,
		# "dependency_parsing": BiaffineParsingHead,
		# "masked_lm": BertStyleMaskedLMHead,
		# "causal_lm": CausalLMHead,
	}

	def add_classification_head(
		self,
		head_name,
		num_labels=2,
		layers=2,
		activation_function="tanh",
		overwrite_ok=False,
		multilabel=False,
		id2label=None,
		use_pooler=False,
	):
		"""
		Adds a sequence classification head on top of the model.
		Args:
			head_name (str): The name of the head.
			num_labels (int, optional): Number of classification labels. Defaults to 2.
			layers (int, optional): Number of layers. Defaults to 2.
			activation_function (str, optional): Activation function. Defaults to 'tanh'.
			overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
			multilabel (bool, optional): Enable multilabel classification setup. Defaults to False.
		"""

		if multilabel:
			head = MultiLabelClassificationHead(
				self, head_name, num_labels, layers, activation_function, id2label, use_pooler
			)
		else:
			head = ClassificationHead(self, head_name, num_labels, layers, activation_function, id2label, use_pooler)
		self.add_prediction_head(head, overwrite_ok)

# if __name__ == "__main__":
# 	config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
# 	bottleneck_adapter = Adapter("bottleneck_adapter", input_size=1024, down_sample=512, config=config)
# 	x = torch
# 	print(bottleneck_adapter)

if True:
	emo_labels = ['Angry', 'Happy', 'Neutral', 'Surprise', 'Sad']
	num_labels = len(emo_labels)
	label2id, id2label = dict(), dict()
	for i, label in enumerate(emo_labels):
		label2id[label] = str(i)
		id2label[str(i)] = label


	# config.add_adapter
	# config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")

	config = Wav2Vec2Config.from_pretrained(
		"jonatasgrosman/wav2vec2-large-xlsr-53-english",
		num_labels=5,
		# add_adapter=True,
		label2id = label2id,
		id2label = id2label
	)
	processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

	test_list = utils.get_file_list("/data/yingting/ESD/en/", mode="test")
	test_wav,  test_labels, sample_rate = utils.read_wav(test_list)
	test_set = EsdDataset(test_wav, test_labels, processor, sample_rate)
	test_loader = DataLoader(test_set, batch_size=64)

	model = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", config=config)
	print(model)
	breakpoint()

	for inputs in test_loader:
		output = model(inputs.input_values)

	
	print(model)

	# model = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", config=config)
	# model.add_adapter("rotten_tomatoes")

	# # Add a matching classification head
	# model.add_classification_head(
	#	 "rotten_tomatoes",
	#	 num_labels=2,
	#	 id2label={ 0: "👎", 1: "👍"}
	#   )
	# # Activate the adapter
	# model.train_adapter("rotten_tomatoes")




