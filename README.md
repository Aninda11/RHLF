# RHLF
Welcome to Advanced Topics in Artificial Intelligence and Machine Learning! This assignment will give you hands-on experience in building and evaluating an RLHF Pipeline with open-source Large Language Models (LLMs).Implement a Reinforcement Learning with Human Feedback (RLHF) model and evaluate the model’s performance on open-source datasets 

# Assignment 1 – RLHF Mini Project

This repository contains two notebook-based RLHF-style experiments implemented with open-source language models, open preference datasets, and the TRL framework.

## Notebooks

### `RHLF1.ipynb`
Branch 1 of the project.

- **Base model:** Qwen2.5-0.5B-Instruct  
- **Baseline training:** SFT on Alpaca-cleaned  
- **Preference dataset:** HH-RLHF-style  
- **Stages included:** SFT baseline, reward model training, DPO alignment, evaluation, and result analysis  

### `RHLF2.ipynb`
Branch 2 of the project.

- **Base model:** SmolLM2-360M-Instruct  
- **Baseline training:** SFT on Alpaca-cleaned  
- **Preference dataset:** HelpSteer2-binarized  
- **Stages included:** SFT baseline, reward model training, DPO alignment, evaluation, and result analysis  

## How to Run

These notebooks were developed and tested in **Google Colab** with a **T4 GPU**.

1. Open the notebook in Google Colab  
2. Enable GPU runtime  
3. Run the cells from top to bottom  

Install the required libraries in Colab:

```python
!pip -q install transformers datasets accelerate peft trl bitsandbytes sentencepiece pandas matplotlib
