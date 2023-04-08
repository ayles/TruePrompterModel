import datasets
import evaluate
import json
import numpy as np
import os
import re
import torch
import torch.onnx
from .phoneme_remapper import PhonemeRemapper
from dataclasses import dataclass
from phonemizer.backend import BACKENDS
from phonemizer.separator import Separator
from torchaudio.models.wav2vec2.utils import import_huggingface_model
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer, Wav2Vec2Config
from typing import Dict, List, Union


def add_command(subparsers):
    parser = subparsers.add_parser('wav2vec2', help='train speech-to-text model')
    parser.add_argument('-d', '--data', required=True, help='path to timit dataset')
    parser.add_argument('-o', '--out', required=True, help='model output dir')
    parser.set_defaults(func=lambda args: train(args.config, args.data, args.out))


def train(config_path, data_path, output_path):
    with open(config_path) as file:
        config = json.load(file)

    remapper = PhonemeRemapper(phonemes=config["phonemes"], mapping=config["mapping"])

    # Return actual dataset from here
    def prepare_dataset():
        backend = BACKENDS["espeak"]("en-us")
        timit = datasets.load_dataset("timit_asr", data_dir=data_path)
        timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id", "file"])

        def remove_special_characters(batch):
            batch["text"] = re.sub('[\,\?\.\!\-\;\:\"]', '', batch["text"]).lower()
            return batch

        timit = timit.map(remove_special_characters)

        def map_to_phonemes(batch):
            res = backend.phonemize(
                batch["text"],
                separator=Separator(phone='|'),
                strip=True,
            )

            for i in range(len(res)):
                words = res[i].split(" ")
                for j in range(len(words)):
                    phonemes = remapper.remap(words[j].split("|"))
                    words[j] = "".join(phonemes)
                res[i] = " ".join(words)

            batch["text"] = res
            return batch

        timit = timit.map(map_to_phonemes, batched=True)

        vocab = dict()
        vocab["<pad>"] = len(vocab)
        vocab["<unk>"] = len(vocab)
        vocab["|"] = len(vocab)
        vocab = vocab | {v: k + len(vocab) for k, v in enumerate(list(remapper.phonemes))}

        return timit, vocab


    dataset, vocab = prepare_dataset()

    result_path = os.path.join(output_path, "exported")
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    try:
        os.mkdir(result_path)
    except FileExistsError:
        pass

    # Save vocabulary for future use
    vocab_path = os.path.join(result_path, "vocab.json")
    with open(vocab_path, 'w', encoding="utf-8") as vocab_file:
        json.dump(vocab, vocab_file, ensure_ascii=False)


    # Here initialization stuff goes
    tokenizer = Wav2Vec2CTCTokenizer(vocab_path)

    sampling_rate = 16000
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=sampling_rate, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
    config.vocab_size = len(vocab)
    config.ctc_loss_reduction = "mean"
    config.pad_token_id = processor.tokenizer.pad_token_id
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        config=config,
    )
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()


    # Need to apply processors on data
    def preprocess_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched" to ensure mapping is correct
        res = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["text"])
        batch["input_values"] = res.input_values[0]
        batch["labels"] = res.labels
        batch["input_length"] = len(batch["input_values"])

        return batch

    max_input_length_in_sec = 4.0
    dataset = dataset.map(preprocess_dataset, remove_columns=dataset.column_names["train"], num_proc=4)
    dataset["train"] = dataset["train"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])


    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
        """

        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


    training_args = TrainingArguments(
      output_dir=output_path,
      group_by_length=True,
      per_device_train_batch_size=8,
      evaluation_strategy="steps",
      num_train_epochs=30,
      fp16=True,
      save_steps=500,
      eval_steps=500,
      logging_steps=500,
      learning_rate=1e-4,
      weight_decay=0.005,
      warmup_steps=1000,
      save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
    )

    trainer.train(resume_from_checkpoint=True)
    model_path = os.path.join(output_path, "model")
    trainer.save_model(model_path)

    model_imported = import_huggingface_model(model)
    model_imported.eval()

    torch.onnx.export(
        model_imported,
        torch.randn(1, 100000, requires_grad=True),
        os.path.join(result_path, "model.onnx"),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch_size",
                1: "samples",
            },
            "output": {
                0: "batch_size",
                1: "samples",
            },
        },
    )

    with open(os.path.join(result_path, "config.json"), 'w', encoding="utf-8") as config_file:
        config = {
            "sampling_rate": sampling_rate,
            "inputs_to_logits_ratio": model.config.inputs_to_logits_ratio,
            "vocab": vocab,
        }
        json.dump(config, config_file, ensure_ascii=False)

