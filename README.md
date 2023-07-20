# 大模型训练

# 支持模型

- [Baichuan7B/13B](https://github.com/baichuan-inc/Baichuan-7B)
- [Moss16B](https://github.com/OpenLMLab/MOSS)
- [ChatGLM/ChatGLM2](https://github.com/THUDM/ChatGLM-6B)

# 支持训练方法

## SFT

- Full parameter tuning
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)

# 支持训练数据集

```需要将数据下载到本地读取```

## SFT-Datasets

- [Belle_open_source_500k](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/blob/main/Belle_open_source_0.5M.json)```单轮对话```：包含约50万条由[BELLE](https://github.com/LianjiaTech/BELLE)项目生成的中文指令数据。
- [Firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)```单论对话```：23种常见的中文NLP任务的数据，并且构造了许多与中华文化相关的数据，如对联、作诗、文言文翻译、散文、金庸小说等。对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万

- [Moss-003-sft-data](https://huggingface.co/datasets/YeungNLP/moss-003-sft-data)```多论对话```：由复旦大学MOSS团队开源的中英文多轮对话数据，包含100万+数据

- [Safety-prompts](https://github.com/thu-coai/Safety-Prompts)```单论对话```：包括100k条中文安全场景的prompts和ChatGPT的回复，涵盖了各类安全场景和指令攻击，可以用于全面评测和提升模型的安全性，也可以用于增强模型关于安全方面的知识，对齐模型输出和人类价值观。

  

# **解析**

Instruction tuning: Causal Language Model的语言生成过程

![!generate](./src/clm.gif)



# 文本生成策略

```图片源自```[Huggingface blog](https://huggingface.co/blog/how-to-generate)

## Greedy Search

![greedy_search](./src/greedy_search.png)

## Beam Search

![greedy_search](./src/beam_search.png)

## Top K Sampling 

![top_k_sampling](./src/top_k_sampling.png)

## Top P Sampling

![top_p_sampling](./src/top_p_sampling.png)