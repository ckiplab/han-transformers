# Oldhan Transformers

This project provides oldhan Chinese models to NLP tasks including language modeling, word segmentation and part-of-speech tagging.

## Dependency
* transformers
* pytorch

## Models

We uploaded our models to HuggingFace hub.
* Pretrained models using a masked language modeling (MLM) objective.
    * [ckiplab/oldhan-bert-base-chinese](https://huggingface.co/ckiplab/oldhan-bert-base-chinese)
* Fine-tuned models for downstream NLP tasks.
    * [ckiplab/oldhan-bert-base-chinese-pos](https://huggingface.co/ckiplab/oldhan-bert-base-chinese-pos?) (part-of-speech tagging)
    * [ckiplab/oldhan-bert-base-chinese-ws](https://huggingface.co/ckiplab/oldhan-bert-base-chinese-ws) (word Segmentation)

## Usage
---
### Installation
```bash
pip install transformers
pip install torch
```

### Inference

* Pre-trained Language Model

    You can use [ckiplab/oldhan-bert-base-chinese](https://huggingface.co/ckiplab/oldhan-bert-base-chinese) directly with a pipeline for masked language modeling:

    ```python
    >>> from transformers import pipeline
    >>> unmasker = pipeline('fill-mask', model='ckiplab/oldhan-bert-base-chinese')
    >>> unmasker("黎[MASK]於變時雍。")

    [{'sequence': '黎 民 於 變 時 雍 。',
    'score': 0.14885780215263367,
    'token': 3696,
    'token_str': '民'},
    {'sequence': '黎 庶 於 變 時 雍 。',
    'score': 0.0859643816947937,
    'token': 2433,
    'token_str': '庶'},
    {'sequence': '黎 氏 於 變 時 雍 。',
    'score': 0.027848130092024803,
    'token': 3694,
    'token_str': '氏'},
    {'sequence': '黎 人 於 變 時 雍 。',
    'score': 0.023678112775087357,
    'token': 782,
    'token_str': '人'},
    {'sequence': '黎 生 於 變 時 雍 。',
    'score': 0.018718384206295013,
    'token': 4495,
    'token_str': '生'}]
    ```

* Part-of-Speech (PoS) Tagging

    In PoS tagging, [ckiplab/oldhan-bert-base-chinese-pos](https://huggingface.co/ckiplab/oldhan-bert-base-chinese-pos?) recognizes parts of speech in a given text. The task is formulated as labeling each word with a part of the speech.

    ```python
    >>> from transformers import pipeline
    >>> classifier = pipeline("token-classification", model="ckiplab/oldhan-bert-base-chinese-pos")
    >>> classifier("帝堯曰放勳")

    [{'entity': 'NB1',
    'score': 0.99410427,
    'index': 1,
    'word': '帝',
    'start': 0,
    'end': 1},
    {'entity': 'NB1',
    'score': 0.98874336,
    'index': 2,
    'word': '堯',
    'start': 1,
    'end': 2},
    {'entity': 'VG',
    'score': 0.97059363,
    'index': 3,
    'word': '曰',
    'start': 2,
    'end': 3},
    {'entity': 'NB1',
    'score': 0.9864504,
    'index': 4,
    'word': '放',
    'start': 3,
    'end': 4},
    {'entity': 'NB1',
    'score': 0.9543974,
    'index': 5,
    'word': '勳',
    'start': 4,
    'end': 5}]
    ```