# LLM Distillation: Literature Review and Design Implications

*Prepared for Distil, a task-conditioned platform for turning an open-source teacher into a size-targeted student model or a standalone code tool.*

## Executive summary

Knowledge distillation (KD) transfers a teacher's behavior to a smaller student by training on a softer, richer signal than hard labels alone. The original formulation used the teacher's temperature-scaled class distribution as a dark-knowledge target [1]. For language models, that idea has expanded into several related families: token-level logit matching, sequence-level synthetic supervision, intermediate-representation matching, preference/reward distillation, on-policy generation, and structural compression followed by recovery training. The engineering choice is not simply “which loss?” but “which teacher signal can be generated, stored, and evaluated for this task and budget?”

Distil should therefore treat distillation as a configurable data-and-training pipeline. A user chooses a trusted open-weight teacher, describes a specialized task, and names a student capacity. The platform can generate task-conditioned demonstrations and rationales, train or fine-tune a compatible student, evaluate it against task metrics, and expose either model weights or a reproducible code artifact. Black-box teachers remain useful because high-quality responses can be captured without logits; white-box teachers unlock token probabilities and hidden-state losses when access and licensing permit.

## 1. From classifier KD to language-model compression

### 1.1 Classical knowledge distillation

Hinton, Vinyals, and Dean introduced KD as a way to transfer “dark knowledge” from a large model to a small one [1]. Let teacher and student logits be \(z^T\) and \(z^S\). With temperature \(T\), the soft distributions are:

\[
p_i^T = \operatorname{softmax}(z_i^T/T), \qquad
p_i^S = \operatorname{softmax}(z_i^S/T).
\]

The student minimizes a mixture of hard-label cross-entropy and a softened KL divergence, commonly scaling the latter by \(T^2\). The softened tail probabilities reveal relationships among plausible alternatives that a one-hot label discards. In autoregressive language modeling, the same intuition applies at each next-token position, but the sequence length, exposure bias, and vocabulary make the optimization and storage costs substantially larger.

### 1.2 Compact Transformer encoders

DistilBERT showed that a Transformer encoder can be reduced while retaining much of BERT's capability through a triple objective: language-model distillation, a cosine embedding loss over hidden states, and a supervised masked-language-model loss [2]. The recipe also removed token-type embeddings and used a smaller architecture, illustrating that distillation and architectural simplification are complementary.

TinyBERT made the transfer more explicit across both pre-training and task-specific stages [3]. It matched attention maps, hidden states, and embeddings using learned transformations where dimensions differed. Its two-stage “general distillation” followed by “task distillation” remains a useful template for Distil: learn broad language behavior first, then specialize the same student on the user's task.

MiniLM distilled self-attention relations rather than attempting to align every hidden dimension [4, 5]. This is attractive when teacher and student widths differ: query-key relation matrices and value-relation signals preserve the geometry of attention while reducing architectural coupling.

MobileBERT used a specially designed bottleneck architecture and inverted-bottleneck blocks to make a BERT-like model mobile-friendly, with teacher-guided pretraining and task transfer [6]. The lesson is that a smaller parameter count is not sufficient to predict latency: memory bandwidth, sequence length, operator shape, and deployment hardware belong in the target specification.

## 2. Sequence-level and autoregressive distillation

Token-level KD asks the student to imitate the teacher's distribution at gold prefixes. Kim and Rush proposed sequence-level KD, in which teacher-generated sequences become pseudo-targets and the student is trained with ordinary maximum likelihood on those sequences [7]. This can reduce multimodality and make generation behavior easier to learn, at the cost of inheriting teacher errors and depending on decoding choices.

For a causal LLM, sequence-level data has three practical forms:

1. **Teacher continuation:** prompt plus sampled or decoded answer.
2. **Teacher explanation:** answer plus rationale, critique, or structured intermediate fields.
3. **Teacher preference pair:** better and worse responses, suitable for preference optimization.

The generation policy matters. Greedy or beam outputs are consistent and cheap; temperature/top-p samples cover more behavior but can be noisy. Distil should retain generation metadata (decoding parameters, model revision, prompt template, and safety filters) so a run is reproducible.

## 3. Modern LLM distillation

### 3.1 Reverse-KL and MiniLLM

Traditional forward KL can encourage a student to cover all teacher modes, including low-probability regions. MiniLLM argues that autoregressive distillation benefits from reverse-KL-like behavior, focusing the student on high-probability teacher modes and addressing length and exposure issues with a sequence-level objective [8]. The work is a reminder that the teacher distribution is conditioned on the student's prefix during training; simply matching teacher logits on teacher-forced prefixes may not reflect inference-time behavior.

### 3.2 Generalized and on-policy KD

GKD frames distillation as a generalized objective that can use teacher outputs on student-generated trajectories, rather than only on gold or teacher-generated prefixes [9]. On-policy data reduces the train–test mismatch: the teacher critiques states the student is likely to visit. It may, however, multiply teacher inference cost and can amplify early student mistakes. A practical system can begin with offline sequence KD and enable on-policy refinement for quality-sensitive runs.

### 3.3 DistiLLM and distribution matching

DistiLLM studies efficient LLM distillation with objectives that improve the student distribution beyond plain supervised fine-tuning on teacher answers [10]. Its broader contribution is operational: high-quality teacher data and a well-chosen distributional objective can make a small number of examples useful, while naïve imitation may plateau. Distil should log both data volume and objective configuration so quality comparisons are meaningful.

### 3.4 White-box versus black-box teachers

When logits and hidden states are available, white-box KD can use:

- **Token-logit KL:** match next-token distributions, optionally only on selected vocabulary positions.
- **Hidden-state/attention losses:** align representations or relational attention patterns.
- **Contrastive or margin losses:** preserve ranking among candidate continuations.
- **Speculative decoding signals:** use a large verifier/teacher to train a fast draft model; acceptance rate becomes a deployment-oriented metric [11].

When only an API or text interface is available, black-box/data distillation uses outputs, critiques, labels, rationales, and preference pairs. It is more portable across providers and licensing boundaries, but cannot reproduce probability calibration or hidden features. Distil's “teacher model name” should carry an explicit access mode and artifact policy rather than implying that every listed model exposes logits.

## 4. Synthetic data and black-box supervision

Self-Instruct bootstrapped instruction-following data by asking a language model to generate instructions, inputs, and outputs, filtering invalid examples, and fine-tuning on the result [12]. Alpaca demonstrated the impact of a relatively small instruction dataset generated from an instruction-following teacher [13]. Vicuna showed how conversations derived from ShareGPT could produce a capable chat model, while also highlighting the fragility of evaluation based on model-as-judge and undocumented data pipelines [14].

Orca used detailed explanations of the teacher's reasoning process and system-level instructions to move beyond answer-only imitation [15]. Orca 2 explored teaching smaller models different reasoning strategies rather than copying a single response style [16]. These results motivate task-aware data generation: ask the teacher for a rubric, a solution, edge cases, and a concise final answer, then decide which fields belong in the student's training target.

Zephyr showed a compact alignment pipeline combining distilled supervision with preference optimization (AI feedback and DPO-style training) [17]. DPO itself provides a simple objective for fitting preference pairs without an explicit reward model [18]. For Distil, preference data can be an optional second phase after supervised distillation, especially for tone, refusal behavior, or ranking tasks.

Phi-style work emphasizes that carefully filtered, synthetic “textbook” data can matter more than raw corpus scale for small models [19, 20]. This does not make synthetic data automatically safe: teacher hallucinations, copyright contamination, prompt leakage, and systematic bias can be copied at scale. Generation must be paired with deduplication, PII/safety checks, provenance, and held-out evaluation.

## 5. Task-specific reasoning and tools

“Distilling step-by-step” found that training on teacher-generated rationales can improve student performance, especially when the student receives intermediate reasoning supervision rather than only a final answer [21]. Rationales are not guaranteed to be faithful explanations, so they should be treated as training scaffolds and evaluated for task utility, not as ground-truth traces.

For specialized jobs, the teacher prompt should request a structured contract:

```text
input -> analysis/fields -> answer -> confidence -> verification notes
```

The contract can be rendered as JSONL for model training, with task-specific validators. Code review and extraction tasks benefit from executable or schema validation; medical and legal tasks need expert review, citations, uncertainty labels, and conservative refusal behavior.

Distil's optional **Code tool** output is a deliberate alternative to shipping weights. A teacher can produce a standalone script, CLI, or API wrapper that solves a narrow task. This may be cheaper and more auditable than training a student when the “model” is mostly orchestration, retrieval, deterministic parsing, or calls to approved services. A tool artifact should include its interface, dependencies, tests/fixtures, license and provenance metadata, and a visible warning that generated code requires review.

## 6. Pruning, structural compression, and recovery distillation

Distillation can follow or accompany parameter reduction. Sheared LLaMA pruned a pretrained LLaMA model toward a target configuration and continued training efficiently, demonstrating that structured pruning can produce useful smaller LLMs without training from scratch [22]. Minitron/Nemotron reports a practical compression recipe combining width/depth pruning, distillation, and data selection, with a focus on reducing training cost [23].

The design space includes:

- **Unstructured pruning:** high sparsity, but speedups require sparse kernels and hardware support.
- **Structured width pruning:** remove heads, intermediate neurons, or embedding dimensions; easier deployment, potentially larger quality loss.
- **Depth pruning:** remove layers or use layer dropping; often recovered with distillation.
- **Quantization plus KD:** reduce precision while using teacher losses to recover quality.
- **Low-rank/adapters:** reduce trainable or shipped parameters without creating a fully independent student.

Target “parameter count” should therefore be accompanied by approximate active parameters, precision, context length, and hardware. A 3B dense BF16 model and a 3B 4-bit model are very different products.

## 7. Taxonomy

| Axis | White-box distillation | Black-box/data distillation |
|---|---|---|
| Teacher access | Weights, logits, hidden states, attention, or a local inference server | Text/API outputs, labels, critiques, rationales, or preferences |
| Primary signal | Token KL, feature/attention matching, contrastive scores, speculative signals | SFT on synthetic sequences, rationale supervision, DPO/preference pairs |
| Strength | Rich calibration and internal representations; can be sample-efficient | Works across model families/providers; simple artifact boundary |
| Cost/risk | Requires compatible access, large activation/logit storage, and license review | Teacher inference can dominate cost; errors and biases are copied |
| Architecture coupling | Often high, especially for hidden-state matching | Low; student can use a different tokenizer/model family |
| Best fit in Distil | Curated local/open-weight models with logit access | Fast task prototypes, API teachers, code-tool generation |

| Signal level | Example | Typical objective | Distil artifact |
|---|---|---|---|
| Logit | Per-token teacher probabilities | Temperature-scaled KL / reverse KL | Sharded logits or regenerated supervision |
| Data | Prompt, answer, rationale, critique, preference pair | SFT, preference optimization, sequence KD | Versioned JSONL with provenance |
| Feature | Hidden states, attention relations | MSE/cosine/relational losses | Optional teacher cache and adapter |
| Structure | Pruned heads/layers, quantized weights | Recovery KD + continued pretraining | Targeted student checkpoint |
| Tool | Script, CLI, workflow, tests | Code generation plus validation | Standalone reviewed code artifact |

## 8. Benchmarks and evaluation

No single benchmark captures whether a distilled student is useful for a specialized job. General evaluations such as MMLU [24], BIG-bench [25], HELM [26], and lm-evaluation-harness [27] help with broad regression checks, but they can be noisy, contaminated, or poorly aligned with a user's workflow. Human preference comparisons and model-as-judge scores should be reported with prompts, judge model, ordering controls, and uncertainty.

Each Distil run should keep:

- task-specific held-out examples and rubric scores;
- exact-match/F1/schema validity for extraction;
- pass@k or test execution for code;
- factuality, citation correctness, and refusal/safety checks where relevant;
- latency, throughput, memory, context length, and cost;
- compression ratio and teacher/student quality deltas;
- dataset provenance, filters, model revisions, seeds, and decoding settings.

Loss curves and aggregate scores are useful run telemetry, not a substitute for an acceptance suite. The UI should make both visible and distinguish generated-data quality from student quality.

## 9. Open problems

1. **Faithfulness and error transfer.** Synthetic rationales may be persuasive but wrong; students can make teacher errors more systematic.
2. **Distribution shift.** A student trained on teacher-generated prompts may fail on real user phrasing or long-tail inputs.
3. **Calibration.** Black-box imitation rarely preserves reliable probabilities or uncertainty.
4. **Evaluation validity.** Benchmarks saturate, leak into pretraining, and may reward style over correctness.
5. **Data and license governance.** Model licenses, generated-content terms, PII, copyrighted code, and user prompts need per-run provenance.
6. **Tokenizer and architecture mismatch.** Logit/feature losses are difficult across tokenizers and model families.
7. **Reasoning supervision.** More rationale tokens raise cost and may teach undesirable verbosity or shortcut explanations.
8. **Efficient on-policy training.** Student-rollout distillation improves robustness but is expensive and unstable.
9. **Safety transfer.** Refusal behavior can disappear under aggressive compression or task specialization.
10. **Capacity targeting.** Parameter count alone does not predict quality, latency, memory, or serving cost.

## 10. Mapping the literature to Distil's product design

### Curated teacher list

The first UI should show model family, parameter count, license, context window, and access mode. The proposed list—Llama 3.1 70B/8B, Qwen2.5 72B/14B/7B, Mistral Large/7B, Gemma 2 27B/9B, DeepSeek-V3, Phi-4, and Mixtral 8x7B—mixes dense and mixture-of-experts systems. Display both total and active parameters for MoE models, and require license acknowledgement before training.

### Task-conditioned data generation

The task/topic prompt and presets become a generation specification. The backend can produce teacher answers, rubrics, rationales, counterexamples, and preference pairs, with a validator selected by task type. A run records prompt template, sampling configuration, filter decisions, and dataset hash. This operationalizes Self-Instruct, Orca, Phi-style synthetic data, sequence KD, and step-by-step supervision without hiding their provenance.

### Size-targeted students

The target-size slider should communicate compression ratio and distinguish a true architecture change from LoRA/quantization. A first implementation can route target sizes to compatible student checkpoints, then add structured pruning and recovery KD inspired by Sheared LLaMA and Minitron. Evaluation must report quality alongside parameter count, precision, latency, and memory.

### Model versus code tool

Some specialized tasks are better served by a generated artifact than by a new model. The output toggle can select a distilled checkpoint or a standalone code tool. The latter should expose a file tree, syntax-highlighted code, tests, dependency manifest, and integration instructions. Both outputs share teacher provenance, task contract, evaluation, and review gates.

### Runs and feedback loop

The run state machine—queued, generating data, training, evaluating, complete—maps directly to the offline pipeline. A liquid/flask progress metaphor communicates progress without pretending that one percentage means quality. Run detail should link generated data, metrics, artifacts, and integration snippets, making the UI a reproducibility surface rather than just a job launcher.

## References

[1] Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531. https://arxiv.org/abs/1503.02531  
[2] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv:1910.01108. https://arxiv.org/abs/1910.01108  
[3] Jiao, X. et al. (2020). *TinyBERT: Distilling BERT for Natural Language Understanding*. arXiv:1909.10351. https://arxiv.org/abs/1909.10351  
[4] Wang, W. et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. arXiv:2002.10957. https://arxiv.org/abs/2002.10957  
[5] Wang, W. et al. (2021). *MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers*. arXiv:2012.15828. https://arxiv.org/abs/2012.15828  
[6] Sun, Z. et al. (2020). *MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices*. arXiv:2004.02984. https://arxiv.org/abs/2004.02984  
[7] Kim, Y., & Rush, A. M. (2016). *Sequence-Level Knowledge Distillation*. arXiv:1606.07947. https://arxiv.org/abs/1606.07947  
[8] Gu, Y. et al. (2023). *MiniLLM: Knowledge Distillation of Large Language Models*. arXiv:2306.08543. https://arxiv.org/abs/2306.08543  
[9] Agarwal, R. et al. (2023). *On-Policy Distillation of Language Models*. arXiv:2306.13649. https://arxiv.org/abs/2306.13649  
[10] Ko, J. et al. (2024). *DistiLLM: Towards Streamlined Distillation for Large Language Models*. arXiv:2402.03898. https://arxiv.org/abs/2402.03898  
[11] Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. arXiv:2211.17192. https://arxiv.org/abs/2211.17192  
[12] Wang, Y. et al. (2023). *Self-Instruct: Aligning Language Models with Self-Generated Instructions*. arXiv:2212.10560. https://arxiv.org/abs/2212.10560  
[13] Taori, R. et al. (2023). *Alpaca: A Strong Replication of InstructGPT*. arXiv:2303.16199. https://arxiv.org/abs/2303.16199  
[14] Chiang, W.-L. et al. (2023). *Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* on preliminary evaluations. arXiv:2306.05685. https://arxiv.org/abs/2306.05685  
[15] Mukherjee, S. et al. (2023). *Orca: A Methodology for Training Smaller Language Models with Large Language Models*. arXiv:2306.02707. https://arxiv.org/abs/2306.02707  
[16] Mitra, A. et al. (2023). *Orca 2: Teaching Small Language Models How to Reason*. arXiv:2311.11045. https://arxiv.org/abs/2311.11045  
[17] Tunstall, L. et al. (2023). *Zephyr: Direct Distillation of LM Alignment*. arXiv:2310.16944. https://arxiv.org/abs/2310.16944  
[18] Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv:2305.18290. https://arxiv.org/abs/2305.18290  
[19] Gunasekar, S. et al. (2023). *Textbooks Are All You Need*. arXiv:2306.11644. https://arxiv.org/abs/2306.11644  
[20] Li, Y. et al. (2023). *Textbooks Are All You Need II: phi-1.5 Technical Report*. arXiv:2309.05463. https://arxiv.org/abs/2309.05463  
[21] Hsieh, C.-Y. et al. (2023). *Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes*. arXiv:2305.02301. https://arxiv.org/abs/2305.02301  
[22] Xia, M. et al. (2024). *Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning*. arXiv:2310.06694. https://arxiv.org/abs/2310.06694  
[23] Muralidharan, S. et al. (2024). *Compact Language Models via Pruning and Knowledge Distillation*. arXiv:2407.14679. https://arxiv.org/abs/2407.14679  
[24] Hendrycks, D. et al. (2021). *Measuring Massive Multitask Language Understanding*. arXiv:2009.03300. https://arxiv.org/abs/2009.03300  
[25] Srivastava, A. et al. (2022). *Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models*. arXiv:2206.04615. https://arxiv.org/abs/2206.04615  
[26] Liang, P. et al. (2023). *Holistic Evaluation of Language Models*. arXiv:2211.09110. https://arxiv.org/abs/2211.09110  
[27] Gao, L. et al. (2023). *A Framework for Few-shot Learning in Language Models*. arXiv:2009.07118. https://arxiv.org/abs/2009.07118  
