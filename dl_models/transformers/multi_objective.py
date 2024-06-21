#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# The below ENV declaration was needed for Mac
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput


# Copied from https://huggingface.co/ibm/MoLFormer-XL-both-10pct/blob/7b12d946c181a37f6012b9dc3b002275de070314/modeling_molformer.py#L794
class MolformerClassificationHead(nn.Module):
    """Head for sequence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(
            config.classifier_dropout_prob
            if config.classifier_dropout_prob is not None
            else config.hidden_dropout_prob
        )
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
        if isinstance(config.hidden_act, str):
            self.classifier_act_fn = ACT2FN[config.hidden_act]
        else:
            self.classifier_act_fn = config.hidden_act
        self.skip_connection = config.classifier_skip_connection

    def forward(self, pooled_output):
        hidden_state = self.dense(pooled_output)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.classifier_act_fn(hidden_state)
        if self.skip_connection:
            hidden_state = residual = hidden_state + pooled_output
        hidden_state = self.dense2(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.classifier_act_fn(hidden_state)
        if self.skip_connection:
            hidden_state = hidden_state + residual
        logits = self.out_proj(hidden_state)
        return logits


# Required for implementing a custom PretrainedModel
class MolformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolformerModel`]. It is used to instantiate an
    Molformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Molformer
    [ibm/MoLFormer-XL-both-10pct](https://huggingface.co/ibm/MoLFormer-XL-both-10pct) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 2362):
            Vocabulary size of the Molformer model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`MolformerModel`] or [`TFMolformerModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 768):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embedding_dropout_prob (`float`, *optional*, defaults to 0.2):
            The dropout probability for the word embeddings.
        max_position_embeddings (`int`, *optional*, defaults to 202):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 1536).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        linear_attention_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the linear attention layers normalization step.
        num_random_features (`int`, *optional*, defaults to 32):
            Random feature map dimension used in linear attention.
        feature_map_kernel (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the generalized random features. If string,
            `"gelu"`, `"relu"`, `"selu"`, and `"gelu_new"` ar supported.
        deterministic_eval (`bool`, *optional*, defaults to `False`):
            Whether the random features should only be redrawn when training or not. If `True` and `model.training` is
            `False`, linear attention random feature weights will be constant, i.e., deterministic.
        classifier_dropout_prob (`float`, *optional*):
            The dropout probability for the classification head. If `None`, use `hidden_dropout_prob`.
        classifier_skip_connection (`bool`, *optional*, defaults to `True`):
            Whether a skip connection should be made between the layers of the classification head or not.
        pad_token_id (`int`, *optional*, defaults to 2):
            The id of the _padding_ token.
    Example:
    ```python
    >>> from transformers import MolformerModel, MolformerConfig
    >>> # Initializing a Molformer ibm/MoLFormer-XL-both-10pct style configuration
    >>> configuration = MolformerConfig()
    >>> # Initializing a model from the ibm/MoLFormer-XL-both-10pct style configuration
    >>> model = MolformerModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "molformer"

    def __init__(
        self,
        vocab_size=2362,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=768,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        embedding_dropout_prob=0.2,
        max_position_embeddings=202,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        linear_attention_eps=1e-6,
        num_random_features=32,
        feature_map_kernel="relu",
        deterministic_eval=False,
        classifier_dropout_prob=None,
        classifier_skip_connection=True,
        pad_token_id=2,
        classification_loss_weight=0.5,
        regression_loss_weight=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.embedding_dropout_prob = embedding_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.linear_attention_eps = linear_attention_eps
        self.num_random_features = num_random_features
        self.feature_map_kernel = feature_map_kernel
        self.deterministic_eval = deterministic_eval
        self.classifier_dropout_prob = classifier_dropout_prob
        self.classifier_skip_connection = classifier_skip_connection
        self.classification_loss_weight = classification_loss_weight
        self.regression_loss_weight = regression_loss_weight if regression_loss_weight is not None else 1-classification_loss_weight


# MultiObject Molfromer model
@dataclass
class MultiObjectiveOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MolformerForMultiobjective(PreTrainedModel):
    config_class = MolformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.molformer = AutoModel.from_pretrained(
            config.name_or_path,
            trust_remote_code=True,
            device_map="auto"
        )
        self.regressor = MolformerClassificationHead(self.config, num_labels=1)
        self.classifier = MolformerClassificationHead(self.config, num_labels=2)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            label_regression: Optional[torch.FloatTensor] = None,
            label_classification: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultiObjectiveOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.molformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        y_hat = self.regressor(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        loss_regression = None
        loss_classification = None
        if label_classification is not None and label_regression is not None:
            # move labels to correct device to enable model parallelism
            label_regression = label_regression.to(logits.device)
            label_classification = label_classification.to(logits.device)
            loss_regression = MSELoss()(y_hat.squeeze(), label_regression.squeeze())
            loss_classification = CrossEntropyLoss()(logits.view(-1, 2), label_classification.view(-1))
            loss = self.config.regression_loss_weight * loss_regression + self.config.classification_loss_weight * loss_classification

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultiObjectiveOutput(
            loss=loss,
            logits=(y_hat, logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )