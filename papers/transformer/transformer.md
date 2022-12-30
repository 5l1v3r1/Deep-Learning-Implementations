---
layout: post
title:  "Attention Is All You Need: Transformer"
date:   2022-12-30 19:15:31 +0900
category: Paper Review
tags: [Deep Learning, NLP, Transformer]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Author: Adam Lee  

# Attention Is All You Need: Transformer
[Source Paper](https://arxiv.org/abs/1706.03762)  
12 Jun 2017  
NeurIPS 2017  

![Title](/assets/posts/paper_review/16.transformer/title.png "Title")  

# 1. Core Contributions
![Transformer](/assets/posts/paper_review/16.transformer/1.transformer.webp "Transformer")
Transformer Architecture Using Self-Attention for Sequence to Sequence Tasks (Machine Translation).  

Self-Attention  
Scaled Dot-Product Attention Scoring Function  
$$ softmax(\dfrac{QK^{T}}{\sqrt{d_{k}}})V $$

Strengths 
- Dispensing RNNs
- Parallel Training
- Solve long term memory problem and gradient vanishing

# 2. Background
Attention mechanism was envisioned as an enhancement for encoder-decoder models (Bahdanau et al., 2014).  

Instead of the decoder focusing on the information sequentially it should focus on the information with respect to its relevance. As a result, the input is transformed into a weighted sum.  

query q operates on key value pairs (k,v) . 

Attention is weighing the input based on relevance  
$$ Attention(q, D) = \sum_{i=1} \alpha(q,k_{i})v_{i}$$.  
Where D is the dataset and $$\alpha$$ is the attention function.  

Common practice to softmax the weights and cover with exponent to prevent the weights from being negative  
$$ \alpha(q,k_{i}) = \dfrac{exp(\alpha(q,k_{i}))}{\sum_{j} exp(\alpha(q,k_{j}))} $$

# 3. Introduction
RNNs, LSTMs have been used previously for sequence modeling.  
Sequential computation remained a problem.  
Previous models have used the attention mechanism in conjunction with RNN encoder-decoder models.  
Transformer architecture is proposed using self-attention while completely dispensing RNNs, drawing global dependencies between input and output.  
Also great at parallelization and can reach SOTA in translation as quickly as twelve hours on eight P100 GPUs.  

New SOTA in translation tasks.  
English-French model training took 3.5 days with 8 GPUs.  

# 4. Related Works
Previously CNNs were used to compute hidden representations in parallel to reduce sequential computation.  
However the number of operations required to learn signals from input and output positions grew.  
In the Transformer this is reduced to constant operations.  
And to make up for the reduced effectiveness, Multi-head Attention is proposed.  

# 5. Transformer
## 5-1. Encoder decoder architecture.  
![Transformer](/assets/posts/paper_review/16.transformer/1.transformer.webp "Transformer")  
6 identical layers (attention heads)  
Residual connection around the attention mechanism and the feedforward network, resulting in an addition then layer normalization.  
Output dimension of the encoder is 512. (d=512)   

## 5-2. Attention Mechanism
![Dot Product Attention](/assets/posts/paper_review/16.transformer/2.scaled dot_product attention.png "Dot Product Attention")  
$$ softmax(\dfrac{QK^{T}}{\sqrt{d_{k}}})V $$  
$$ d_{k} $$ division is to ensure the variance of the dot product is one.  
$$ Q \in \mathcal{R}^k$$  
$$ K \in \mathcal{R}^k$$  
$$ d_{k} = k $$  

![Multihead Attention](/assets/posts/paper_review/16.transformer/3.multihead_attention.png "Multihead Attention")  
Each independent attention outputs are concatenated 
