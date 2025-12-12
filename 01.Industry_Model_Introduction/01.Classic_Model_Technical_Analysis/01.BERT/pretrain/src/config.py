import os
import json
import logging
from .utils import load_from_cache
from .utils import get_mindrecord_list
from .utils import get_json_config

# define your mindrecord dataset path
pretrain_mindrecord_list = [
    '/data1/dataset/bert_pretrain/bert/src/generate_mindrecord/9_15_wiki/mr_aa/',
    '/data1/dataset/bert_pretrain/bert/src/generate_mindrecord/9_15_wiki/mr_ab/',
    '/data1/dataset/bert_pretrain/bert/src/generate_mindrecord/9_15_wiki/mr_ac/',
    '/data1/dataset/bert_pretrain/bert/src/generate_mindrecord/9_15_wiki/mr_ad/',
    '/data1/dataset/bert_pretrain/bert/src/generate_mindrecord/9_15_wiki/mr_ae/',
]

class PretrainedConfig:
    """
    Pretrained Config.
    """
    pretrained_config_archive = {}
    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.train_batch_size = kwargs.pop('train_batch_size', 128)
        self.do_save_ckpt = kwargs.pop('do_save_ckpt', True)
        self.jit = kwargs.pop('jit', True)
        self.do_train = kwargs.pop('do_train', True)
        # self.save_ckpt_path = kwargs.pop('save_ckpt_path', os.path.join('')'/data0/bert/outputs/model_save')
        self.save_steps = kwargs.pop('save_steps',10000)
        self.epochs = kwargs.pop('epochs', 40)
        self.lr = kwargs.pop('lr', 1e-5)
        self.warmup_steps = kwargs.pop('warmup_steps', 10000)
        # self.dataset_mindreocrd_dir = kwargs.pop('dataset_mindreocrd_dir',\
        # get_mindrecord_list(pretrain_mindrecord_list))

    @classmethod
    def load(cls, pretrained_model_name_or_path, **kwargs):
        """load config."""
        force_download = kwargs.pop('force_download', False)
        if os.path.exists(pretrained_model_name_or_path):
            # File exists.
            config_file = pretrained_model_name_or_path
        elif pretrained_model_name_or_path in cls.pretrained_config_archive:
            logging.info("The checkpoint file not found, start to download.")
            config_url = cls.pretrained_config_archive[pretrained_model_name_or_path]
            config_file = load_from_cache(pretrained_model_name_or_path + '.json',
                                          config_url, force_download=force_download)
        else:
            # Something unknown
            raise ValueError(
                f"unable to parse {pretrained_model_name_or_path} as a local path or model name")

        config = cls.from_json(config_file)

        return config

    @classmethod
    def from_json(cls, file_path):
        """load config from json."""
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        config_map = json.loads(text)
        config = cls()
        for key, value in config_map.items():
            setattr(config, key, value)
        return config

CONFIG_ARCHIVE_MAP = {
    "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
    "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/config.json",
    "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/config.json",
    "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/config.json",
    "bert-base-multilingual-uncased": "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/config.json",
    "bert-base-multilingual-cased": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/config.json",
    "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/config.json",
    "bert-base-german-cased": "https://huggingface.co/bert-base-german-cased/resolve/main/config.json",
    "bert-large-uncased-whole-word-masking": "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/config.json",
    "bert-large-cased-whole-word-masking": "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/config.json",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/config.json",
    "bert-base-cased-finetuned-mrpc": "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/config.json",
    "bert-base-german-dbmdz-cased": "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/config.json",
    "bert-base-german-dbmdz-uncased": "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese": "https://huggingface.co/cl-tohoku/bert-base-japanese/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-char": "https://huggingface.co/cl-tohoku/bert-base-japanese-char/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking/resolve/main/config.json",
    "TurkuNLP/bert-base-finnish-cased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/config.json",
    "TurkuNLP/bert-base-finnish-uncased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/config.json",
    "wietsedv/bert-base-dutch-cased": "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/config.json",
    "sentence-transformers/all-MiniLM-L6-v2": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json"
    # See all BERT models at https://huggingface.co/models?filter=bert
}

class BertConfig(PretrainedConfig):
    """Configuration for BERT
    """
    pretrained_config_archive = CONFIG_ARCHIVE_MAP
    def __init__(self, bert_config, **kwargs):
        super().__init__(**kwargs)
        bert_config_data = get_json_config(bert_config)
        self.vocab_size = int(bert_config_data['vocab_size'])
        self.hidden_size = bert_config_data['hidden_size']
        self.num_hidden_layers = bert_config_data['num_hidden_layers']
        self.num_attention_heads = bert_config_data['num_attention_heads']
        self.hidden_act = bert_config_data['hidden_act']
        self.intermediate_size = bert_config_data['intermediate_size']
        self.hidden_dropout_prob = bert_config_data['hidden_dropout_prob']
        self.attention_probs_dropout_prob = bert_config_data['attention_probs_dropout_prob']
        self.max_position_embeddings = bert_config_data['max_position_embeddings']
        self.type_vocab_size = int(bert_config_data['type_vocab_size'])
        self.initializer_range = bert_config_data['initializer_range']
        self.layer_norm_eps = 1e-12