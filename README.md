# SMART: Small Reasons, Large Hints

Codebase for the paper:
> **Guiding Reasoning in Small Language Models with LLM Assistance**  
> Yujin Kim, Euiin Yi, Minu Kim, Se-Young Yun, Taehyeon Kim  
> [arXiv:2504.09923](https://arxiv.org/abs/2504.09923)

---

SMART introduces a novel test-time framework where **Small Language Models (SLMs)** reason step-by-step, and **Large Language Models (LLMs)** provide guidance only when necessary. This selective LLM intervention enables lightweight models to achieve up to **98.9% of LLM accuracy while reducing LLM token usage by up to 90%**, making it practical for collaborative settings like on-device + API deployments.

## 🚀 Quickstart

To run a SMART experiment, use:

```bash
bash run_qwen.sh <OPTION_NUMBER>
# or
bash run_llama.sh <OPTION_NUMBER>
```

### Available options (for Qwen):
| OPTION | Model Pair            | Search Type | Score Method |
|--------|------------------------|-------------|---------------|
| 0      | Qwen2.5-7B            | Best-of-N   | PRM           |
| 1      | Qwen2.5-7B            | Beam Search | PRM           |
| 2      | Qwen2.5-7B            | Beam Search | Confidence    |
| 3      | Qwen2.5-1.5B          | Best-of-N   | PRM           |
| 4      | Qwen2.5-7B + SMART    | Best-of-N   | PRM           |
| 5      | Qwen2.5-7B + SMART    | Beam Search | PRM           |

## 📁 Project structure
```
SMART/
├── recipes/                        # YAML configs for each model
│   ├── Llama-3.1-8B-Instruct/
│   ├── Llama-3.2-1B-Instruct/
│   ├── Qwen2.5-1.5B-Instruct/
│   └── Qwen2.5-7B-Instruct/
│   └── launch_array.slurm
├── scripts/                        # Test-time compute logic
├── src/                            # Core implementation
├── run_qwen.sh                     # Shell launcher for Qwen variants
├── run_llama.sh                    # Shell launcher for Llama variants
└── README.md
```

## 📊 Reproducing Results
SMART is evaluated on **MATH500** and extended to **MMLU-Pro**. Details on settings, scoring, and results are in our [paper](https://arxiv.org/abs/2504.09923). SMART supports:
- Best-of-N & Beam Search
- Scoring with PRM or token-level confidence(TLC)

## 📦 Installation
```bash
conda create -n smart python=3.11 && conda activate smart
pip install -e '\.\[dev\]'  # install SMART in editable mode
huggingface-cli login
sudo apt-get install git-lfs
```

## 📚 Citation
If you find this repository useful, please cite:
```bibtex
@article{kim2025guiding,
  title={Guiding Reasoning in Small Language Models with LLM Assistance},
  author={Kim, Yujin and Yi, Euiin and Kim, Minu and Yun, Se-Young and Kim, Taehyeon},
  journal={arXiv preprint arXiv:2504.09923},
  year={2025}
}
```