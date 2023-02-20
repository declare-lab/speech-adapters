import torch
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from transformers.adapters import AdapterConfig
from transformers.adapters.model_mixin  import EmbeddingAdaptersWrapperMixin
from transformers.adapters.heads import (ModelWithFlexibleHeadsAdaptersMixin,
										 MultiLabelClassificationHead,
										  ClassificationHead)
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING, BertModel, BertPreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward

from transformers.adapters import BertAdapterModel, PfeifferInvConfig, PrefixTuningConfig, LoRAConfig

from transformers.adapters.context import AdapterSetup

'''
@add_start_docstrings(
    """Bert Model transformer with the option to add multiple flexible heads on top.""",
    BERT_START_DOCSTRING,
)
class BertAdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        self._init_head_modules()

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head=None,
        output_adapter_gating_scores=False,
        output_adapter_fusion_attentions=False,
        **kwargs
    ):
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_adapter_gating_scores=output_adapter_gating_scores,
            output_adapter_fusion_attentions=output_adapter_fusion_attentions,
        )
        # BERT & RoBERTa return the pooled output as second item, we don't need that in these heads
        if not return_dict:
            head_inputs = (outputs[0],) + outputs[2:]
        else:
            head_inputs = outputs
        pooled_output = outputs[1]

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
'''




if __name__ == "__main__":
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	sentence = "It's also, clearly, great fun."
	input_data = tokenizer(sentence, return_tensors="pt")

	model = BertAdapterModel.from_pretrained("bert-base-uncased")
	# model_origin = BertModel.from_pretrained("bert-base-uncased")
	
	

	# print("------>>> Trainable params(before add adapter):", sum(p.numel() for p in model_origin.parameters() if p.requires_grad))
	# print("------>>> Trainable params(before add adapter):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	# config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
	# model.add_adapter("bottleneck_adapter", config=config)
	# config = PrefixTuningConfig(flat=False, prefix_length=30)
	# model.add_adapter("prefix_tuning", config=config)
	# config = LoRAConfig(r=8, alpha=16)
	# model.add_adapter("lora_adapter", config=config)
	# print(model)
	# for n,p in model.named_parameters():
	# 	if "lora_adapter" in n:
	# 		pass
	# 	elif "prefix_tuning" in n:
	# 		pass
	# 	else:
	# 		p.requires_grad=False

	# print("------>>> Trainable params(after freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))

	# for name, param in model.named_parameters():
	# 	if param.requires_grad:
	# 		print(name, param.requires_grad, param.size())

	# config = PfeifferInvConfig()
	# model.add_adapter("lang_adapter", config=config)
	config = PrefixTuningConfig(flat=False, prefix_length=30)
	model.add_adapter("prefix_tuning", config=config)
	model.train_adapter("prefix_tuning")
	model.set_active_adapters("prefix_tuning")
	# config = LoRAConfig(r=8, alpha=16)
	# model.add_adapter("lora_adapter", config=config)
	# output = model(input_data)
	# output = model_origin(**input_data)

	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name, param.requires_grad, param.size())
	exit()
	output = model(**input_data)
	# print(model)
	# output = model(input_data["input_ids"])

