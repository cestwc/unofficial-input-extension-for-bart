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

from extended_model import BartForSequenceClassificationWithInputsEmbeddings
bart = BartForSequenceClassificationWithInputsEmbeddings.from_pretrained('facebook/bart-base')

class Discriminator(nn.Module):
	def __init__(self, bart):

		super().__init__()

		self.bart = bart

		self.embedding = nn.Embedding(bart.model.shared.num_embeddings, bart.model.shared.embedding_dim)

		self.embedding.weight.data.copy_(bart.model.shared.weight.data)

	def forward(self, text):

		#text = [batch size, sent len]

		embedded = self.embedding(text)

		return self.bart(input_ids = text, inputs_embeds = embedded).logits
		
model = Discriminator(bart)
```

Around 30M more paramaters will be introduced by your own embedding layer.


