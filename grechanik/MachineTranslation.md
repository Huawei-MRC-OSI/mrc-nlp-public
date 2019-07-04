
## Competitions
* WMT (Workshop on Statistical Machine Translation, now it seems to be called Conference on Machine
  Translation). <http://www.statmt.org/wmt19/>
  Each year they provide a set of tasks, like:
  - News translation (for several language pairs)
  - Automatic Post-Editing
  - Some domain translation (IT, Medical, etc)
  - Quality estimation
  They evaluate the submitted translations both manually and automatically.
  Training data is mainly taken from European Parliament Proceedings
  <http://www.statmt.org/europarl/>.
  Interestingly, commercial online translators participate against their will (but are anonymized).

* IWSLT

## Leaderboards
The most popular datasets on paperswithcode are WMT2014 English-French and WMT2014 English-German.
Both are scored with BLEU. Links to paperswithcode:
- <https://paperswithcode.com/task/machine-translation>
- <https://paperswithcode.com/task/unsupervised-machine-translation>

* WMT2014 English-French
  1. Transformer Big + BT 45.6 Understanding Back-Translation at Scale 2018
  2. Local Joint Self-attention 43.3 Joint Source-Target Self Attention with Locality Constraints 2019
  3. Transformer Big 43.2 Scaling Neural Machine Translation 2018
  4. DynamicConv 43.2 Pay Less Attention with Lightweight and Dynamic Convolutions 2019
  5. LightConv 43.1 Pay Less Attention with Lightweight and Dynamic Convolutions 2019
  6. Transformer (big) + Relative Position Representations 41.5 Self-Attention with Relative
     Position Representations 2018
  7. Weighted Transformer (large) 41.4 Weighted Transformer Network for Machine Translation 2017

* WMT2014 English-German
  1. DynamicConv 29.7 Pay Less Attention with Lightweight and Dynamic Convolutions 2019
  2. Transformer Big 29.3 Scaling Neural Machine Translation 2018
  3. Evolved Transformer Big 29.3 The Evolved Transformer 2019
  4. Transformer (big) + Relative Position Representations 29.2 Self-Attention with Relative
     Position Representations 2018
  5. Transformer Big with FRAGE 29.11 FRAGE: Frequency-Agnostic Word Representation 2018
  6. LightConv 28.9 Pay Less Attention with Lightweight and Dynamic Convolutions 2019
  7. Weighted Transformer (large) 28.9 Weighted Transformer Network for Machine Translation 2017

* **Unsupervised** WMT2016 English-German (Note that it is WMT2016, not WMT2014)
  1. MASS 28.3 MASS: Masked Sequence to Sequence Pre-training for Language Generation 2019
  2. SMT + NMT 26.9 An Effective Approach to Unsupervised Machine Translation 2019
  3. MLM pretraining for encoder and decoder 26.4 Cross-lingual Language Model Pretraining 2019
     (looks like this work made a big leap in BLEU)
  4. SMT as posterior regularization 21.7 Unsupervised Neural Machine Translation with SMT as
     Posterior Regularization 2019
  5. PBSMT + NMT 20.2 Phrase-Based & Neural Unsupervised Machine Translation 2018
  6. Synthetic bilingual data init 20.0 Unsupervised Neural Machine Translation Initialized by
     Unsupervised Statistical Machine Translation 2018

## Papers

* Understanding Back-Translation at Scale, 2018, Edunov et al, Facebook, Google Brain
  <https://arxiv.org/abs/1808.09381v2>
  - They use Back-Translation, i.e. first train a system on bilingual data, then generate more data
    by translating some monolingual data.
  - There several ways of generating the output sequence: beam search, sampling, top-k sampling.
    They tried all this stuff and also tried adding noise (dropping words). Beam with noise seems to
    be beeter (not sure, not universally).

* Pay Less Attention with Lightweight and Dynamic Convolutions, 2019, Wu et al, Cornell, Facebook
  <https://arxiv.org/abs/1901.10430v2>
  - They propose some new convolutions that are better than attention. These new convolutions are
    Lightweight Convolutions and Dynamic Convolutions.
  - LightConv are based on Depthwise convolutions. Depthwise convolutions perform a convolution
    independently over every channel. LightConv additionally tie weights between some channels and
    also the weights are normalized with softmax across the temporal dimension.
  - They believe that LightConv may be faster than they are now when they are implemented as a CUDA
    kernel.
  - They also use Gated Linear Units in their architecture
  - DynamicConv is a LightConv whose weights are computed dynamically. Currently they use a linear
    layer to compute these weights. (If you remember about the softmax, this makes it quite similar
    to attention).
  - Seems like they keep attention between the encoder and the decoder, but I'm not sure.

* Scaling Neural Machine Translation, 2018, Ott et al, Facebook
  <https://arxiv.org/pdf/1806.00187.pdf>
  "This paper shows that reduced precision and large batch training can speedup training by nearly
  5x on a single 8-GPU machine" (65% by using reduced fp precision, 40% by increasing batch size and
  lr)

* A blog post on BLEU criticism and alternatives
  <https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213>

* Cross-lingual Language Model Pretraining, 2019, Lample and Conneau, Facebook
  <https://arxiv.org/pdf/1901.07291v1.pdf>
  - They train a BERT-like model on a BERT-like lm task, however there are some differences:
    - They do not pass pairs of sentences, just a bunch of consequent sentences
    - They choose the word to mask with some probability which is dependent on its frequency.
    - They also have a similar task for parallel texts (concatenate parallel sentences and mask)
    - It's multilingual, each piece of text comes from a random language, and also they share a
      common dictionary of tokens.
  - Turned out it is very important to pretrain the encoder, pretraining the decoder is not that
    important.
  - The rest of the Unsupervised MT approach is taken from the paper Lample et al, Phrase-Based &
    Neural Unsupervised Machine Translation

* Phrase-Based & Neural Unsupervised Machine Translation, 2018, Lample et al, Facebook
  <https://arxiv.org/pdf/1804.07755.pdf>
  - The approach to unsupervised MT is as follows: train language models for the two languages with
    a shared encoder, and then simply use back-translation.
  - They train the tokenizer jointly on both corpora.
  - They share both the encoder and the decoder, the decoder receives an additional token indicating
    the target language, but to the encoder the language information is not provided. Sharing the
    encoder is critial, sharing the decoder is not.
  - They also pretrain token embeddings with fastText. This turned out to be important.
  - They try two neural language model architectures: LSTM and 4-layer transformer.
  - lm are trained as something like denoising autoencoder.
  - Additionally they also try some classical method which is initialized based on embeddings. They
    refer to some other paper. For language modeling in this case they also use some non-neural
    approach
  - The classical approach won this time.

## Projects

* There is OpenNMT, some open-source translation toolkit <http://opennmt.net/>.
  There are two github repos, <https://github.com/OpenNMT/OpenNMT-py>, ~3000 stars, and
  <https://github.com/OpenNMT/OpenNMT-tf> ~750 stars.
  There is a paper <https://www.aclweb.org/anthology/P17-4012> and an updated paper
  <https://www.aclweb.org/anthology/W18-1817>.

* <https://github.com/pytorch/fairseq> Facebook AI Research Sequence-to-Sequence Toolkit. Contains
  implementations of some models.

## What does WMT2014 look like?

Like sentence-aligned short news articles (so there is context). (This is the test set, I haven't
looked at the training data).

```
<doc docid="1074-mk" genre="news" origlang="ru">
<seg id="1">Fines for unauthorized street vending in Vologda to increase severalfold</seg>
<seg id="2">Municipal authorities believe that increased fines will help bring order to Vologda and free the streets from spontaneous booths and stalls.</seg>
<seg id="3">The majority of street vendors do not pay taxes, and the products they sell frequently do not pass health inspections.</seg>
<seg id="4">This means that the health of the citizens of Vologda is exposed to danger</seg>
<seg id="5">Therefore, the minimum fine imposed on citizens for unauthorized vending will increase sixfold beginning in November, from 500 rubles to 3,000 rubles, reports the press relations service of the Administration of Vologda.</seg>
</doc>
```

```
<doc docid="1074-mk" genre="news" origlang="ru">
<seg id="1">Штрафы за несанкционированную уличную торговлю в Вологде вырастут в разы</seg>
<seg id="2">По мнению городских властей, увеличение размеров штрафов поможет навести порядок в Вологде и освободить улицы от стихийных палаток и лотков.</seg>
<seg id="3">Большинство уличных торговцев не платят налоги, продукция, которую они реализуют, зачастую не проходит соотвествующий санитарный контроль.</seg>
<seg id="4">А это значит, что под угрозу ставится здоровье вологжан.</seg>
<seg id="5">Так, нижний предел штрафа для граждан за несанкционированную торговлю с ноября возрастет в шесть раз - с 500 рублей до 3000 рублей, сообщает пресс-служба Администрации Вологды.</seg>
</doc>
```

