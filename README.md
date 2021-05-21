# unofficial-input-extension-for-bart
to allow using inputs embeddings in Hugging Face BartForSequenceClassification

## Usage
```
!pip install transformers
!rm -rf unofficial-input-extension-for-bart; git clone https://github.com/cestwc/unofficial-input-extension-for-bart.git; cp unofficial-input-extension-for-bart/* ./
```

```python
import torch
import torch.nn as nn

from extended_model import BartForSequenceClassificationWithSoftInputs

model = BartForSequenceClassificationWithSoftInputs.from_pretrained('facebook/bart-base')

predictions = model(input_ids = your_input_ids, inputs_probs = your_inputs_probs).logits
```

Models trained with this architecture could be later be loaded as normal BartForSequenceClassification, as they share the same parameters.
