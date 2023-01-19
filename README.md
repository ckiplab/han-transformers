# Han Transformers

This project provides ancient Chinese models to NLP tasks including language modeling, word segmentation and part-of-speech tagging.

Our paper has been accepted to ROCLING. Please check out our [paper](https://aclanthology.org/2022.rocling-1.21/).

## Dependency
* transformers &le; 4.15.0
* pytorch

## Models

We uploaded our models to HuggingFace hub.
* Pretrained models using a masked language modeling (MLM) objective.
    * [ckiplab/bert-base-han-chinese](https://huggingface.co/ckiplab/bert-base-han-chinese)
* Fine-tuned models for word Segmentation.
    * [ckiplab/bert-base-han-chinese-ws](https://huggingface.co/ckiplab/bert-base-han-chinese-ws) (Merge)
    * [ckiplab/bert-base-han-chinese-ws-shanggu](https://huggingface.co/ckiplab/bert-base-han-chinese-ws-shanggu) (上古)
    * [ckiplab/bert-base-han-chinese-ws-zhonggu](https://huggingface.co/ckiplab/bert-base-han-chinese-ws-zhonggu) (中古)
    * [ckiplab/bert-base-han-chinese-ws-jindai](https://huggingface.co/ckiplab/bert-base-han-chinese-ws-jindai) (近代)
    * [ckiplab/bert-base-han-chinese-ws-xiandai](https://huggingface.co/ckiplab/bert-base-han-chinese-ws-xiandai) (現代)
* Fine-tuned models for part-of-speech tagging.
    * [ckiplab/bert-base-han-chinese-pos](https://huggingface.co/ckiplab/bert-base-han-chinese-pos?) (Merge)
    * [ckiplab/bert-base-han-chinese-pos-shanggu](https://huggingface.co/ckiplab/bert-base-han-chinese-pos-shanggu) (上古 / [標記列表](shanggu.md))
    * [ckiplab/bert-base-han-chinese-pos-zhonggu](https://huggingface.co/ckiplab/bert-base-han-chinese-pos-zhonggu) (中古 / [標記列表](zhonggu.md))
    * [ckiplab/bert-base-han-chinese-pos-jindai](https://huggingface.co/ckiplab/bert-base-han-chinese-pos-jindai) (近代 / [標記列表](jindai.md))
    * [ckiplab/bert-base-han-chinese-pos-xiandai](https://huggingface.co/ckiplab/bert-base-han-chinese-pos-xiandai) (現代 / [標記列表](xiandai.md))


## Training Corpus
The copyright of the datasets belongs to the Institute of Linguistics, Academia Sinica.
* [中央研究院上古漢語標記語料庫](http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/akiwi/kiwi.sh)
* [中央研究院中古漢語語料庫](http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/dkiwi/kiwi.sh)
* [中央研究院近代漢語語料庫](http://lingcorpus.iis.sinica.edu.tw/cgi-bin/kiwi/pkiwi/kiwi.sh)
* [中央研究院現代漢語語料庫](http://asbc.iis.sinica.edu.tw)


## Usage

### Installation
```bash
pip install transformers==4.15.0
pip install torch==1.10.2
```

### Inference

* **Pre-trained Language Model**

    You can use [ckiplab/bert-base-han-chinese](https://huggingface.co/ckiplab/bert-base-han-chinese) directly with a pipeline for masked language modeling.

    ```python
    from transformers import pipeline

    # Initialize 
    unmasker = pipeline('fill-mask', model='ckiplab/bert-base-han-chinese')

    # Input text with [MASK]
    unmasker("黎[MASK]於變時雍。")
    
    # output
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

    You can use [ckiplab/bert-base-han-chinese](https://huggingface.co/ckiplab/bert-base-han-chinese) to get the features of a given text in PyTorch.
    
    ```python
    from transformers import AutoTokenizer, AutoModel

    # Initialize tokenzier and model
    tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-han-chinese")
    model = AutoModel.from_pretrained("ckiplab/bert-base-han-chinese")

    # Input text
    text = "黎民於變時雍。"
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    # get encoded token vectors
    output.last_hidden_state    # torch.Tensor with Size([1, 9, 768])

    # get encoded sentence vector
    output.pooler_output        # torch.Tensor with Size([1, 768])
    ```

* **Word Segmentation (WS)**

    In WS, [ckiplab/bert-base-han-chinese-ws](https://huggingface.co/ckiplab/bert-base-han-chinese-ws) divides written the text into meaningful units - words. The task is formulated as labeling each word with either beginning (B) or inside (I).

    ```python
    from transformers import pipeline

    # Initialize
    classifier = pipeline("token-classification", model="ckiplab/bert-base-han-chinese-ws")

    # Input text
    classifier("帝堯曰放勳")

    # output
    [{'entity': 'B',
    'score': 0.9999793,
    'index': 1,
    'word': '帝',
    'start': 0,
    'end': 1},
    {'entity': 'I',
    'score': 0.9915047,
    'index': 2,
    'word': '堯',
    'start': 1,
    'end': 2},
    {'entity': 'B',
    'score': 0.99992275,
    'index': 3,
    'word': '曰',
    'start': 2,
    'end': 3},
    {'entity': 'B',
    'score': 0.99905187,
    'index': 4,
    'word': '放',
    'start': 3,
    'end': 4},
    {'entity': 'I',
    'score': 0.96299917,
    'index': 5,
    'word': '勳',
    'start': 4,
    'end': 5}]
    ```

* **Part-of-Speech (PoS) Tagging**

    In PoS tagging, [ckiplab/bert-base-han-chinese-pos](https://huggingface.co/ckiplab/bert-base-han-chinese-pos) recognizes parts of speech in a given text. The task is formulated as labeling each word with a part of the speech.

    ```python
    from transformers import pipeline

    # Initialize
    classifier = pipeline("token-classification", model="ckiplab/bert-base-han-chinese-pos")

    # Input text
    classifier("帝堯曰放勳")

    # output
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


## Model Performance

### Pre-trained Language Model, **Perplexity &darr;**

<table>
  <tr>
    <th rowspan="2">Trained Model</th>
    <th rowspan="2">MLM Training Data</th>
    <th colspan="4">MLM Testing Data</th>
  </tr>
  <tr>
    <th>上古</th>
    <th>中古</th>
    <th>近代</th>
    <th>現代</th>
  </tr>
  <tr>
    <td rowspan="5">ckiplab/bert-base-Chinese</td>
    <td style="text-align: center;">上古</td>
    <td class="right bold"><strong>24.7588</strong></td>
    <td class="right">87.8176</td>
    <td class="right">297.1111</td>
    <td class="right">60.3993</td>
  </tr>
  <tr>
    <td style="text-align: center">中古</td>
    <td class="right">67.861</td>
    <td class="right">70.6244</td>
    <td class="right">133.0536</td>
    <td class="right">23.0125</td>
  </tr>
  <tr>
    <td style="text-align: center">近代</td>
    <td class="right">69.1364</td>
    <td class="right">77.4154</td>
    <td class="right bold"><strong>46.8308</strong></td>
    <td class="right">20.4289</td>
  </tr>
  <tr>
    <td style="text-align: center">現代</td>
    <td class="right">118.8596</td>
    <td class="right">163.6896</td>
    <td class="right">146.5959</td>
    <td class="right">4.6143</td>
  </tr>
  <tr>
    <td style="text-align: center">All</td>
    <td class="right">31.1807</td>
    <td class="right bold"><strong>61.2381</strong></td>
    <td class="right">49.0672</td>
    <td class="right bold"><strong>4.5017</strong></td>
  </tr>
  <tr>
    <td>ckiplab/bert-base-chinese</td>
    <td style="text-align: center">-</td>
    <td class="right">233.6394</td>
    <td class="right">405.9008</td>
    <td class="right">278.7069</td>
    <td class="right">8.8521</td>
  </tr>
</table>
<!-- <style>
    th,td {
        text-align: center;
    }
    .right {
        text-align: right;
    }
    .bold {
        font-weight: bold;
    }
</style> -->

### Word Segmentation (WS), **F1 score &uarr;**
<table>
  <tr>
    <th rowspan="2">Trained Model</th>
    <th rowspan="2">WS Training Data</th>
    <th colspan="4">WS Testing Data</th>
  </tr>
  <tr>
    <th>上古</th>
    <th>中古</th>
    <th>近代</th>
    <th>現代</th>
  </tr>
  <tr>
    <td rowspan="5">ckiplab/bert-base-Chinese<BR>w/ finetune on all period MLM</td>
    <td style="text-align: center">上古</td>
    <td class="right"><strong>0.9761</strong></td>
    <td class="right">0.8857</td>
    <td class="right">0.8329</td>
    <td class="right">0.7038</td>
  </tr>
  <tr>
    <td style="text-align: center">中古</td>
    <td class="right">0.9264</td>
    <td class="right"><strong>0.9265</strong></td>
    <td class="right">0.8948</td>
    <td class="right">0.7838</td>
  </tr>
  <tr>
    <td style="text-align: center">近代</td>
    <td class="right">0.9087</td>
    <td class="right">0.9219</td>
    <td class="right"><strong>0.9465</strong></td>
    <td class="right">0.8121</td>
  </tr>
  <tr>
    <td style="text-align: center">現代</td>
    <td class="right">0.8702</td>
    <td class="right">0.8358</td>
    <td class="right">0.8494</td>
    <td class="right"><strong>0.9694</strong></td>
  </tr>
  <tr>
    <td style="text-align: center">All</td>
    <td class="right">0.9745</td>
    <td class="right bold">0.92</td>
    <td class="right">0.941</td>
    <td class="right">0.9673</td>
  </tr>
  <tr>
    <td>ckiplab/bert-base-chinese-ws</td>
    <td style="text-align: center">-</td>
    <td class="right">0.8657</td>
    <td class="right">0.8291</td>
    <td class="right">0.8432</td>
    <td class="right"><strong>0.9813</strong></td>
  </tr>
</table>

## License
[<img src="https://www.gnu.org/graphics/gplv3-with-text-136x68.png">
](https://www.gnu.org/licenses/gpl-3.0.html)

Copyright (c) 2022 [CKIP Lab](https://ckip.iis.sinica.edu.tw/) under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.html).

## Citation
Please cite our paper if you use Han-Transformers in your work:

```bibtex
@inproceedings{lin-ma-2022-hantrans,
    title = "{H}an{T}rans: An Empirical Study on Cross-Era Transferability of {C}hinese Pre-trained Language Model",
    author = "Lin, Chin-Tung  and  Ma, Wei-Yun",
    booktitle = "Proceedings of the 34th Conference on Computational Linguistics and Speech Processing (ROCLING 2022)",
    year = "2022",
    address = "Taipei, Taiwan",
    publisher = "The Association for Computational Linguistics and Chinese Language Processing (ACLCLP)",
    url = "https://aclanthology.org/2022.rocling-1.21",
    pages = "164--173",
}
```