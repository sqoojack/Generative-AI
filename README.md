# Generative AI – Spring 2025 @ NYCU

This is the course repository for **Generative AI**, offered in Spring 2025 at **National Yang Ming Chiao Tung University (NYCU)**.  


## Lab 1: Prompt Engineering
Use prompt engineering to design API from **gemini 2.0-flash** and use it to answer the muiltple choice question in specific subject.
### My prompt technique: 
1. Use CoT(Chain Of Thought)
2. Few shot prompting 
3. Clear and concise.
4. Use the "task" field to narrow down the scope, making it easier to find an answer.

### What I learned:
- How to implement APIs in code.
- How to disign prompts to help model achieve better perfromance.

## Lab 2: Fine-tuning an LLM using QLoRA
The purpose of this project is to fine-tune the Mistral-7B model to generate paper abstracts using the QLoRA approach. The process includes data splitting, model training, evaluation, and inference. The following steps outline the complete workflow.

### What I learned:
- How to fine-tune a large language model(LLM) and understand the concept of QLoRA.
- The meaning and impact of hyperparameters like **top-p, top-k, temperature,** and **LoRA configuration.**
- How to apply **4-bit quantization** to save GPU memory when resources are limited.
- The importance of choosing an appropriate model from HuggingFace-selecting the wrong model can lead to poor performance.


## Lab3: Using RAG to improve the accuracy of LLMs
Implemented document-level Q&A using RAG with a provided dataset. Designed effective prompts to query the LLM and built a pipeline to generate accurate answers.

### What I learned:
- What is RAG and how to implement it.
- What's the different from fine-tune and RAG.
