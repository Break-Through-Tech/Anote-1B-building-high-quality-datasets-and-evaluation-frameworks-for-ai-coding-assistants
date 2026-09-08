# Building High-Quality Datasets and Evaluation Frameworks for AI Coding Assistants

**Break Through Tech AI Studio — Fall 2026 · Host Company: Anote**

> 📌 **Status:** Milestone 1 — Foundations & Data. Sections below marked
> _To be completed_ will be filled in as the work happens.

---

## 👥 Team Members

| Name | GitHub Handle | Pronouns |
|------|---------------|----------|
| Luke Wang | | He/Him |
| Brandon Romero | [@romero-brandon](https://github.com/romero-brandon) | He/Him |
| Sabad Modi | [@SabadModi](https://github.com/SabadModi) | He/Him |
| Sophia Zhang | [@sophia-z-23](https://github.com/sophia-z-23) | She/Her |
| Angela Liu | [@AngelaLiu07](https://github.com/AngelaLiu07) | She/Her |
| Phyo Thu Kha | [@levithukha](https://github.com/levithukha) | He/Him |

**Challenge Advisor:** Natan Vidra, Anote ([@nv78](https://github.com/nv78))  
**AI Studio Coach:** Bhavya Gopal ([@bhavya-gopal](https://github.com/bhavya-gopal))

---

## 🏗️ Project Overview

### Break Through Tech AI connection

This is a Fall 2026 **AI Studio Challenge Project**. AI Studio pairs teams of Break Through
Tech AI Fellows with an industry host company for a semester-long, real-world ML project,
supported by a Challenge Advisor from the company and an AI Studio Coach.

### Host company and objective

Our host company is **Anote**, which builds AI solutions for coding, debugging, and code
understanding.

Per the project brief, we will use code datasets, developer interactions, and code
evaluation benchmarks alongside large language models and prompt-based evaluation
techniques to build and evaluate an AI system that improves code generation, debugging,
and code understanding performance. This addresses Anote's challenge of **improving the
accuracy, reliability, and evaluation of AI-powered coding assistants in real-world
development environments**.

### Scope guardrails

Per SME review, the project is scoped to stay stable on Google Colab's free tier:

| In scope | Out of scope |
|----------|--------------|
| Prompt engineering and retrieval | Model fine-tuning |
| Static evaluation | Live sandboxed code execution |

---

## 🗓️ Project Milestones

| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Foundations & Data | Define project scope · curate or construct a dataset (code snippets + prompts, ground-truth outputs) · data preprocessing and formatting · establish baseline models · define evaluation metrics |
| **October** | Modeling & Evaluation | Build evaluation pipelines (functional correctness, code similarity / semantic correctness scoring) · run experiments comparing zero-shot vs. adapted models · analyze performance gaps and failure cases |
| **November** | Optimization & Integration | Improve dataset quality (edge cases, error cases, difficult prompts) · iterate on model and evaluation pipeline · integrate outputs into a simple interface · produce final benchmarking report and demo |

### Final outcome (December)

- A working evaluation framework for AI coding models
- A curated dataset for code tasks (generation, fixing, testing)
- Demonstrated improvement vs. baseline models
- A reproducible pipeline for benchmarking coding assistants

---

## 📊 Dataset

| | |
|---|---|
| **Name** | Terminal-Bench 2.0 (Harbor Framework public dataset) |
| **Source** | [GitHub](https://github.com/harbor-framework/terminal-bench-2-1/tree/main/tasks) (source of truth) · [Hugging Face](https://huggingface.co/datasets/harborframework/terminal-bench-2.0) (easier download for Colab) |
| **Format** | CSV, JSON |
| **Size** | Under 1 GB |

- **Terminal-Bench** is a benchmark and execution harness for evaluating how well AI agents
  perform real, end-to-end tasks in a sandboxed terminal environment (compiling code,
  debugging, resolving security issues), maintained by the Harbor project. Terminal-Bench
  2.0 is a curated set of 89 hard tasks, each with a unique environment, a human-written
  solution, and verification tests.
- **[Harbor](https://github.com/harbor-framework/harbor)** is the companion framework used
  to run and score agents against the benchmark. It supports evaluating agents such as
  Claude Code, OpenHands, and Codex CLI.
- Related repos worth reviewing for structure and format:
  [terminal-bench-2-1](https://github.com/harbor-framework/terminal-bench-2-1) (current
  verified task set) and
  [frontier-bench](https://github.com/harbor-framework/frontier-bench) (successor benchmark).

---

## 📐 Evaluation Metrics

As defined in the project brief:

**Quantitative**

- **Pass@1** — does the top generated candidate pass a fixed, pre-defined test case set
  (evaluated statically rather than through a live execution loop)
- **Structural / semantic correctness** via standard NLP similarity metrics — CodeBLEU,
  exact match, AST diff
- **Accuracy of bug fixes / code transformations** against reference solutions
- **Reduction in error rate** vs. baseline model outputs

**Qualitative**

- Code readability and quality
- Developer usefulness — does the output solve the task effectively

---

## 👩🏽‍💻 Setup and Installation

### Prerequisites

Python 3.10+ and Git. Notebooks are intended to run on Google Colab's free tier.

### Team Colab notebook

Shared working notebook: [`Anote1B.ipynb`](https://colab.research.google.com/drive/1p11xmZyjo8Bq1-chwih-L-K2LJrRt9hQ)
(access is restricted to the team).

### Local setup

```bash
# 1. Clone the repository
git clone https://github.com/Break-Through-Tech/Anote-1B-building-high-quality-datasets-and-evaluation-frameworks-for-ai-coding-assistants.git
cd Anote-1B-building-high-quality-datasets-and-evaluation-frameworks-for-ai-coding-assistants

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# then edit .env and add your keys
```

### API keys

Baseline model comparisons call hosted LLM APIs. Copy `.env.example` to `.env` and fill in
the providers being used. `.env` is gitignored — **never commit an API key**. If one is
pushed, revoke it immediately; deleting the commit is not enough.

### Getting the dataset

Download the Hugging Face mirror into `data/raw/`:

```bash
pip install huggingface_hub
huggingface-cli download harborframework/terminal-bench-2.0 \
  --repo-type dataset --local-dir data/raw/terminal-bench-2.0
```

Dataset files are gitignored — see [`data/README.md`](data/README.md).

### Running the pipeline

_To be completed._

---

## 📁 Repository Structure

```
├── data/                 # Datasets (contents gitignored — see data/README.md)
│   ├── raw/              # Downloaded task files, unmodified
│   └── processed/        # Cleaned / formatted data produced by our code
├── notebooks/            # Jupyter notebooks
├── src/                  # Reusable pipeline code
├── results/              # Metric outputs, figures, reports
├── requirements.txt      # Python dependencies
├── .env.example          # Template for API keys (copy to .env)
└── Challenge-Project-Overview.md   # Brief from our Challenge Advisor
```

---

## 📊 Data Exploration

_To be completed._ Will cover dataset exploration and preprocessing approaches, insights
from EDA, challenges and assumptions encountered, and supporting visualizations.

---

## 🧠 Model Development

_To be completed._ Will cover the models used, prompt and retrieval strategies, and the
training / evaluation setup.

---

## 📈 Results & Key Findings

_To be completed._ Will report performance against the metrics above, how the models
performed, and insights from evaluating model fairness.

---

## 🚀 Next Steps

_To be completed._ Stretch goals identified in the project brief:

- Build a real-time evaluation system integrated into the Anote coding assistant
- Add reinforcement learning / feedback loops (RLHF or RLAIF)
- Expand to multi-file or full-repo reasoning
- Add code execution environments (sandbox testing)
- Create a leaderboard comparing models (Claude, GPT, open-source)
- Incorporate human-in-the-loop feedback pipelines

---

## 📝 License

_To be decided with our Challenge Advisor._

---

## 📄 References

**Papers**

- Terminal-Bench 2.0 — the benchmark and dataset used in this project. [arXiv:2601.11868](https://arxiv.org/abs/2601.11868)
- *Evaluating Large Language Models Trained on Code* (Codex / HumanEval) — introduces the Pass@k metric. [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
- *CodeBLEU: a Method for Automatic Evaluation of Code Synthesis*. [arXiv:2009.10297](https://arxiv.org/abs/2009.10297)
- *Out of the BLEU: How Should We Assess Quality of Code Generation Models?* — on why match-based metrics can diverge from human judgment. [arXiv:2208.03133](https://arxiv.org/abs/2208.03133)

**Documentation**

- [Harbor Framework](https://github.com/harbor-framework/harbor) · [Harbor docs — running Terminal-Bench](https://www.harborframework.com/docs/tutorials/running-terminal-bench)
- [terminal-bench-2-1](https://github.com/harbor-framework/terminal-bench-2-1) · [frontier-bench](https://github.com/harbor-framework/frontier-bench)

**Talks**

- [Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in CLI](https://www.youtube.com/watch?v=RRFeCml3mrQ)
- [Benchtalks #1: Alex Shaw on Terminal-Bench & Harbor](https://www.youtube.com/watch?v=UCn5gG0haCI)

---
## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.