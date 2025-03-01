import os
from typing import Union
from transformers import PretrainedConfig, GPTNeoXConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LlavaPythiaVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPVisionModel`]. It is used to instantiate a
    CLIP vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        mm_vision_select_feature (`str`, *optional*, defaults to `"patch"`):
            The feature to select from the vision encoder output. Can be one of `"patch"` or `"cls_patch"`.
        mm_vision_select_layer (`int`, *optional*, defaults to `-2`):
            The layer to select from the vision encoder output.

    Example:

    ```python
    >>> from transformers import CLIPVisionConfig, CLIPVisionModel

    >>> # Initializing a CLIPVisionConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPVisionConfig()

    >>> # Initializing a CLIPVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava_pythia_clip_vision_model"

    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            projection_dim=512,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=32,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            mm_vision_select_feature="patch",
            mm_vision_select_layer=-2,
            vision_model_name_or_path="clip",
            concat="None",
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_vision_select_layer = mm_vision_select_layer
        self.vision_model_name_or_path = vision_model_name_or_path
        self.concat = concat

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "llava_pythia":
            config_dict = config_dict["vision_config"]["vision_tower"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ProjectorConfig(PretrainedConfig):
    model_type = "llava_pythia_projector"

    def __init__(
            self,
            mm_projector_type="linear",
            mm_hidden_size=768,
            hidden_size=2560,
            **kwargs
    ):
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "llava_pythia":
            config_dict = config_dict["vision_config"]["mm_projector"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)



from typing import List

# for initialize act head 

DEFAULT_VISUAL_CONFIG = {
    "vision_tower": LlavaPythiaVisionConfig().to_dict(),
    "mm_projector": ProjectorConfig().to_dict(),
}

# print(DEFAULT_ACT_CONFIG['act'])

class LlavaPythiaConfig(GPTNeoXConfig):
    """
    Configuration class for LlavaPythia.
    Extends GPTNeoXConfig and includes additional parameters for the vision tower and multimodal projector.
    """

    model_type = "llava_pythia"

    def __init__(
            self,
            vocab_size=50432,
            hidden_size=2048,
            max_position_embeddings=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=None,
            hidden_act="gelu",
            rotary_pct=1.0,
            rotary_emb_base=10000,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            classifier_dropout=None,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            bias=False,
            use_cache=True,
            bos_token_id=0,
            eos_token_id=1,
            tie_word_embeddings=False,
            use_parallel_residual=True,
            vision_config=None,
            vision_tower=None,
            mm_projector=None,
            mm_vision_select_layer=-2,  # default to the penultimate layer
            mm_vision_select_feature="patch",  # default to the patch features
            mm_use_im_start_end=False,
            mm_use_im_patch_token=False,
            mm_projector_type="mlp2x_gelu",
            action_head_type="fc",
            action_dim=10,
            state_dim=8,
            chunk_size=16,
            # VLA-Cache相关配置
            use_cache_mechanism=False,  # 是否启用VLA-Cache机制
            cache_size=0.3,  # 缓存大小占总Token数的比例
            importance_threshold=0.7,  # Token重要性阈值
            cache_update_freq=10,  # 缓存更新频率
            cache_warmup_steps=5,  # 缓存预热步数
            **kwargs
    ):
        if vision_config is None:
            self.vision_config = DEFAULT_VISUAL_CONFIG
        else:
            self.vision_config = vision_config

        self.concat = "None"
        super().__init__(**kwargs)


if __name__ == "__main__":
    print(LlavaPythiaVisionConfig())
