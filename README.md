# Overview

The case study I would be working on is a research paper named &quot;Neural Machine Translation for Low-Resourced Indian Languages&quot; which is written by students from DTU, Delhi.

So, as we all know we don&#39;t have good translators from English language to our mother tongues. If there are even available, the performance of those is not as good as the translators of English to Spanish or even English to Hindi, there can be multiple reasons for this -

- Obvious one is that these are less spoken languages.
- English to Indian language translation poses the challenge of morphological and structural divergence. For instance, 
      (i) the number of parallel corpora and 
      (ii) differences between languages, mainly the morphological richness and variation in word order due to syntactical divergence. English has Subject-Verb-Object (SVO) whereas Tamil and Malayalam have Subject-Object-Verb (SOV).
- They used old or conventional technologies and models like Rule-based machine translation (RBMT) or corpus based as these approaches had their own flaws.
- One more reason for the bad performance of these translators is not using the new approaches and models that have been introduced in recent years which are going to throw the old models out of the window. I am talking about models like Bert or sequence to sequence models which use bidirectional LSTMs, attention layers, Byte-Pair-Encoding (BPE).

What these guys have tried to incorporate

- The main problem for Indian mother tongue languages is very less dataset or i will say clean parallel corpora for these languages was not available. So, these guys have cleaned and pre-processed the data for these languages-

![](RackMultipart20210624-4-188dojs_html_6323c002dbbeccce.png)

- These guys have created models for languages
  - English - Tamil
  - English - Malayalam
- The model and algorithms they used
  - sequence to sequence models with LSTM layers
  - Word Embeddings
  - Pre-trained Byte-Pair-Encoding
  - Multihead self-attention layers

So, in this case study I am going to first replicate their and then she what new things I can add into this to get better bleu scores or seven try with different metric than bleu score

# ![](RackMultipart20210624-4-188dojs_html_237499165a11f2b9.gif)

# Research-Papers/Solutions/Architectures/Kernels

1. [[2004.13819] Neural Machine Translation for Low-Resourced Indian Languages (arxiv.org)](https://arxiv.org/abs/2004.13819) - This is the same paper that my case study is based on and also my overview.
2. [https://www.appliedaicourse.com/lecture/11/applied-machine-learning-online-course/4150/attention-models-in-deep-learning/8/module-8-neural-networks-computer-vision-and-deep-learning](https://www.appliedaicourse.com/lecture/11/applied-machine-learning-online-course/4150/attention-models-in-deep-learning/8/module-8-neural-networks-computer-vision-and-deep-learning) - this is the link from the Applied AI live session for attention based models, in this they have explained the drawbacks of general sequence to sequence models, why should we use attention models and how they works.

**Overview** -

![](RackMultipart20210624-4-188dojs_html_fa391bffbaabbd18.png)

From the traditional seq2seq models after the encoder part we receive a w vector which acts as an input to our decoder hidden state. So the drawback for this is if input sentences are lengthy then the vector w wouldn&#39;t be able to capture the essence of the input sentence.

![](RackMultipart20210624-4-188dojs_html_9c08c03aa607b3e.png)

whereas attention models instead of receiving a vector from the last layer of encoder they receive a context vector, which is basically a weighted sum of outputs of the encoder. So, these weights help in focusing on a particular part of the input sentence.

1. [https://bpemb.h-its.org/](https://bpemb.h-its.org/) - BPEmb is a collection of pre-trained subword embeddings in 275 languages, based on Byte-Pair Encoding (BPE) and trained on Wikipedia. Its intended use is as input for neural models in natural language processing.
2. [https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern-nlp-eb36c7df4f10](https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern-nlp-eb36c7df4f10) -Byte Pair Encoding - A simple data compression algorithm first introduced in 1994 supercharging almost all advanced NLP models of today (including BERT).

![](RackMultipart20210624-4-188dojs_html_ec6167b5545ac07a.png)

1. [EnTam: An English-Tamil Parallel Corpus (EnTam v2.0) (cuni.cz)](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1454)

[OPUS - an open source parallel corpus (nlpl.eu)](https://opus.nlpl.eu/)

[UMC005: English-Urdu Parallel Corpus (cuni.cz)](https://ufal.mff.cuni.cz/umc/005-en-ur/)

These links have corpus for languages Tamil, Malayalam and Urdu languages.

1. [https://www.tensorflow.org/tutorials/text/transformer](https://www.tensorflow.org/tutorials/text/transformer) - this an example of multi head attention model with code explanation.

![](RackMultipart20210624-4-188dojs_html_237499165a11f2b9.gif)

# First Cut Approach

\*\*\* Explain in steps about how you want to approach this problem and the initial experiments that you want to do. _**(MINIMUM 200 words)**_ \*\*\*

My first cut solution would be totally based on the applied AI live session of attention model. After seeing the results and analyzing the issues with it then I will try to incorporate new things from the paper and other material.

Steps -

1. Dataset - I have downloaded dataset from this link - [https://drive.google.com/drive/folders/1b3h13rBwTOZRygT6ZIdk4eZ9MKmXSZJa](https://drive.google.com/drive/folders/1b3h13rBwTOZRygT6ZIdk4eZ9MKmXSZJa) ,it&#39;s already cleaned, just need to do preprocessing.
2. Tokenize of data - I am using TensorFlow tokenization
3. Creating a data pipeline for the data
4. Using a sequence-to-sequence model with a attention layer and a SoftMax layer at the end.

Few important points that i need help with-

1. The length of 99.9% for English and Malayalam sentences is around 70, isn&#39;t that too long.
2. Dictionary size for English and Malayalam is around 54041 and 272131 and when I went through the number of times a word occurs in a sentence, around 30% of words occur only 1 time. So, are there any chances we can remove them without compromising with model performance?
