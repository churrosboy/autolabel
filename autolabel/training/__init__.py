"""LoRA fine-tuning of an open VLM as a CoC auto-labeler.

Design
------
* base model      : Qwen3-VL Instruct (2B on a laptop, 4B/8B on a GPU box).
* adaptation      : LoRA (PEFT) on the language-model projections only.
* extra head      : one small MLP on the hidden state of the last prompt token
                    that classifies the longitudinal and lateral decision
                    (Table 1 vocabulary + "none").  It gives a structured
                    prediction the filter can compare with the generated
                    sentence, and it is an auxiliary loss that pushes the model
                    to commit to one canonical decision.
* objective       : L = CE(text) + head_weight * (CE(lon) + CE(lat))
* data            : an Autolabel project directory (windows.jsonl + labels.jsonl
                    + frames/) or an export directory (samples.jsonl + frames/).

Modules: data.py (samples, split, head labels), collate.py (processor ->
tensors), model.py (AutolabelModel), train.py (loop, eval, checkpoints),
fetch_frames.py (physical_ai_av frames for imported D3D windows).
torch is imported lazily so the rest of Autolabel stays light.
"""
