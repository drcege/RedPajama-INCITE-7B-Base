---
license: apache-2.0
language:
- en
---

# RedPajama-Base-INCITE-6.9B

RedPajama-Base-INCITE-6.9B-v1, is a large transformer-based language model developed by Together Computer and trained on the RedPajama-Data-1T dataset.

## Model Details
- **Developed by**: Together Computer.
- **Model type**: Language Model
- **Language(s)**: English
- **License**: Apache 2.0
- **Model Description**: A 6.9B parameter pretrained language model.

# Quick Start

## GPU Inference

This requires a GPU with 8GB memory.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
# init
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-Base-INCITE-6.9B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-Base-INCITE-6.9B-v1", torch_dtype=torch.float16)
model = model.to('cuda:0')
# infer
inputs = tokenizer("Hello", return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.8)
output_str = tokenizer.decode(outputs[0])
print(output_str)
```

## GPU Inference in Int8

This requires a GPU with 6GB memory.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
# init
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-Base-INCITE-6.9B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-Base-INCITE-6.9B-v1", device_map="auto", load_in_8bit=True)
# infer
inputs = tokenizer("Hello", return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.8)
output_str = tokenizer.decode(outputs[0])
print(output_str)
```

## CPU Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
# init
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-Base-INCITE-6.9B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-Base-INCITE-6.9B-v1", torch_dtype=torch.bfloat16)
# infer
inputs = tokenizer("<human>: Hello!\n<bot>:", return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.8)
output_str = tokenizer.decode(outputs[0])
print(output_str)
```


# Uses

## Direct Use 

The model is intended for research purposes. Possible research areas and tasks include

- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of dialogue models or language models.
- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.
- Research on dialogue models or language models.

Excluded uses are described below.

### Misuse, Malicious Use, and Out-of-Scope Use

It is the responsibility of the end user to ensure that the model is used in a responsible and ethical manner.

#### Out-of-Scope Use

RedPajama-Base-INCITE-6.9B is a language model and may not perform well for other use cases outside of its intended scope. 
For example, it may not be suitable for use in safety-critical applications or for making decisions that have a significant impact on individuals or society. 
It is important to consider the limitations of the model and to only use it for its intended purpose.

#### Misuse and Malicious Use

RedPajama-Base-INCITE-6.9B is designed for language modeling.
Misuse of the model, such as using it to engage in illegal or unethical activities, is strictly prohibited and goes against the principles of the OpenChatKit community project.

Using the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:

- Generating fake news, misinformation, or propaganda
- Promoting hate speech, discrimination, or violence against individuals or groups
- Impersonating individuals or organizations without their consent
- Engaging in cyberbullying or harassment
- Defamatory content
- Spamming or scamming
- Sharing confidential or sensitive information without proper authorization
- Violating the terms of use of the model or the data used to train it
- Creating automated bots for malicious purposes such as spreading malware, phishing scams, or spamming

## Limitations

RedPajama-Base-INCITE-6.9B, like other language models, has limitations that should be taken into consideration. 
For example, the model may not always provide accurate or relevant answers, particularly for questions that are complex, ambiguous, or outside of its training data. 
We therefore welcome contributions from individuals and organizations, and encourage collaboration towards creating a more robust and inclusive chatbot.

## Training

**Training Data**

Please refer to [togethercomputer/RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)

**Training Procedure**

- **Hardware:** TODO @Dan
- **Optimizer:** 
- **Gradient Accumulations**: 
- **Num of Tokens:** 800B Tokens
- **Learning rate:** 

## Community

Join us on [Together Discord](https://discord.gg/6ZVDU8tTD4)