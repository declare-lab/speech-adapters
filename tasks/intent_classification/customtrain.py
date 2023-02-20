import torch
import numpy as np
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs.get("labels")
		# forward pass
		outputs = model(**inputs)
		
		intent_logits = outputs.get("logits")

		intent_loss = 0
		start_index = 0
		predicted_intent = []
		
		values_per_slot = [6,14,4]
		loss_fct = nn.CrossEntropyLoss().to(labels.device)

		
		for slot in range(3):
			end_index = start_index + values_per_slot[slot]
			subset = intent_logits[:, start_index:end_index]

			# breakpoint()

			intent_loss += loss_fct(subset, labels[:, slot])
			predicted_intent.append(subset.max(1)[1])

			# breakpoint()
			
			start_index = end_index

		def idx2slots(indices: torch.Tensor):
			action_idx, object_idx, location_idx = indices.cpu().tolist()
			return (
				self.Sy_intent["action"][action_idx],
				self.Sy_intent["object"][object_idx],
				self.Sy_intent["location"][location_idx],
			)

		return (intent_loss, outputs) if return_outputs else intent_loss

def compute_metrics(eval_pred):

	action = eval_pred.predictions[:, :6].argmax(axis=1)
	object_ = eval_pred.predictions[:, 6:20].argmax(axis=1)
	location = eval_pred.predictions[:, 20:].argmax(axis=1)

	predicted_intent = np.vstack((action, object_, location)).T

	acc_list = (predicted_intent == eval_pred.label_ids).prod(1).astype(np.float32).tolist()

	acc = sum(acc_list) * 1.0 / len(acc_list)
	
	return {"acc":acc}