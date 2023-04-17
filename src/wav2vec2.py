import datasets
import evaluate
import json
import numpy as np
import os
import re
import torch
import torch.onnx
import tempfile
from .remapper import PhonemeRemapper
from .phonemizer import EpitranPhonemizer
from dataclasses import dataclass
from torchaudio.models.wav2vec2.utils import import_huggingface_model
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer, Wav2Vec2Config
from typing import Dict, List, Union


sampling_rate = 16000


def _prepare_vocabulary(remapper):
    vocab = dict()
    vocab["<pad>"] = len(vocab)
    vocab["<unk>"] = len(vocab)
    vocab["|"] = len(vocab)
    for phoneme in remapper.vocabulary:
        if phoneme not in vocab:
            vocab[phoneme] = len(vocab)
    return vocab


def _prepare_dataset(languages, split, samples, phonemizer, remapper):
    def preprocess_dataset(language, dataset):
        dataset = dataset.remove_columns(list(set(dataset.column_names).difference({ "sentence", "audio" })))

        def remap(batch):
            res = phonemizer.phonemize(language, [sentence for sentence in batch["sentence"]])
            ret = []
            for sentences in res:
                sentence = []
                for phonemes in sentences:
                    phonemes = remapper.remap(phonemes)
                    sentence.append("".join(phonemes))
                ret.append(" ".join(sentence))
            batch["sentence"] = ret
            return batch

        global sampling_rate
        dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=sampling_rate))
        return dataset.map(remap, batched=True)

    res_datasets = []
    for language in languages:
        dataset = datasets.load_dataset("mozilla-foundation/common_voice_13_0", language, split=split, streaming=True).shuffle(seed=7).take(samples)
        dataset = preprocess_dataset(language, dataset)
        res_datasets.append(dataset)
    return datasets.concatenate_datasets(res_datasets)


def _prepare_dataset_generator(languages, split, samples, phonemizer, remapper):
  def f():
    train = _prepare_dataset(languages, split, samples, phonemizer, remapper)
    for item in train:
      yield item
  return f


def train(languages, remapper_path, output_path):
    with open(remapper_path) as file:
        remapper = PhonemeRemapper.load(file)

    phonemizer = EpitranPhonemizer()

    vocab = _prepare_vocabulary(remapper)
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, 'w') as f:
            json.dump(vocab, f, ensure_ascii=False)
        tokenizer = Wav2Vec2CTCTokenizer(tmp.name)

    global sampling_rate
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

    if True:
        train = datasets.Dataset.from_generator(_prepare_dataset_generator(languages, "train", 3000, phonemizer, remapper))
        test = datasets.Dataset.from_generator(_prepare_dataset_generator(languages, "test", 800, phonemizer, remapper))


        # Need to apply processors on data
        def preprocess_dataset(batch):
            audio = batch["audio"]

            # batched output is "un-batched" to ensure mapping is correct
            res = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["sentence"])
            batch["input_values"] = res.input_values[0]
            batch["labels"] = res.labels
            batch["input_length"] = len(batch["input_values"])
            batch["labels_length"] = len(batch["labels"])

            return batch


        max_input_length_in_sec = 6.0
        train = train.map(preprocess_dataset, remove_columns=train.column_names, num_proc=12)
        test = test.map(preprocess_dataset, remove_columns=test.column_names, num_proc=12)

        train = train.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

        train = train.filter(lambda x: x > 0, input_columns=["labels_length"])
        test = test.filter(lambda x: x > 0, input_columns=["labels_length"])

        train.save_to_disk(os.path.join(output_path, "saved_train"))
        test.save_to_disk(os.path.join(output_path, "saved_test"))

    train = datasets.load_from_disk(os.path.join(output_path, "saved_train"))
    test = datasets.load_from_disk(os.path.join(output_path, "saved_test"))

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
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=40,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        weight_decay=0.005,
        learning_rate=1e-4,
        warmup_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()


def add_command(subparsers):
    parser = subparsers.add_parser('wav2vec2', help='train speech-to-text model')
    parser.add_argument('-r','--remapper', required=True, help='path to remapper config file')
    parser.add_argument('-o','--output', required=True, help='path to output folder')
    parser.add_argument('-l','--language', required=True, nargs='+', help='language to build vocabulary to, space-separated if many')
    parser.set_defaults(func=lambda args: train(args.language, args.remapper, args.output))

