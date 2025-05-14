ARC-Challenge/
├── deepseek_first_finetune_3_epoch/
│   └── complete_script.py/
├── Transduction/
│   └── global_script.py/
├── test_time_training/
│   └── ttt.py/ 
├── judge_gemini.py        
├── judge_llm.py           


| Script path                                       | Phase / purpose                           |
|---------------------------------------------------|-------------------------------------------|
| `deepseek_first_finetune_3_epoch/complete_script.py` | **1 ➜ Initial fine-tune** on DeepSeek-Chat |
| `Transduction/global_script.py`                   | **2 ➜ Global transduction** post-process   |
| `test_time_training/ttt.py`                       | **3 ➜ Test-time training** (per-task tweak)|
| `judge_gemini.py`                                 | Gemini pipeline for ARC evaluation        |
| `judge_llm.py`                                    | DeepSeek pipeline for ARC evaluation      |
| `data/training/*.json`                            | ARC task files                            |

---

## 🖥️  Requirements
```bash
python >= 3.9
pip install google-generativeai openai python-dotenv requests
