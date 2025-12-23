# LLM-Action-Item-Extractor

## Overview

Meetings often produce unstructured notes that make it easy for action items to be missed or forgotten. This project builds an LLM-based information extraction pipeline that converts raw meeting transcripts into structured action items, including the task, owner, and deadline.

The system uses a LoRA-adapted large language model to reliably extract action items from noisy, multi-speaker conversations and output them in a strict JSON schema suitable for downstream automation and analytics.

## Key Features

- Extracts action items from real meeting transcripts
- Identifies task, owner, and deadline, including implicit references
- Outputs schema-validated JSON for reliability
- Uses parameter-efficient fine-tuning (LoRA) to improve consistency over prompt-only baselines
- Designed for internal analytics and workflow automation, not generic summarization

## Evaluation

The system is evaluated on its ability to produce **strict, schema-valid JSON outputs**, measured using Pydantic validation.

### Schema Validity Results

| Model | Schema Validity |
|------|----------------|
| Prompt-only baseline (Mistral-7B-Instruct) | 0% |
| LoRA-fine-tuned model (this project) | **97.14%** |

**Metric:** Percentage of outputs that strictly match the required JSON schema  
**Dataset:** Held-out real + synthetic meeting transcripts  
**Validation:** Automated schema checks using Pydantic

## Why This Project

Many commercial meeting tools provide high-level summaries but do not expose structured outputs, evaluation metrics, or customization. This project focuses on:

- Transparent data creation and labeling
- Measurable extraction performance
- Engineering reliability for downstream systems

The result is a production-oriented NLP pipeline that mirrors how action-item extraction is built and evaluated in real-world ML teams.
