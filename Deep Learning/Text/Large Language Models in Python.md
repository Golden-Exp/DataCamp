#datacamp #deeplearning 
# Large Language Models
```python
from transformers import pipeline
sentiment_classifier = pipeline("text-classification")
outputs = sentiment_classifier("""Dear Seller ... blah blah""")
print(outputs)
```
```
[{'labels':'positive', 'score':0.999}]
```

pipeline imports necessary LLMs and its pre-trained weights for the task required(here it's text classification) and we can just use it for ourselves - default classes for text classification is positive and negative

to get a specific model that you want, we use the model parameter
```python
classifier = pipeline("text-classification",
					  model="nlptown/bert-base-multilingual-uncased-sentiment")
```

Language tasks using LLMs can be of two types 
1. Language Generation
2. Language Understanding
![[Pasted image 20240311190325.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240311190325.png/?raw=true)

#### Text classification
here, the given text is classified into categories that are already pre-defined. Sentiment analysis is an example of this.
```python
llm = pipeline("text-classification")
text = "blah blah"
outputs = llm(text)
print(outputs[0]["label"])
```

#### Text Generation
here, a text based on the maximum length is generated to complete a given prompt.
```python
llm = pipeline("text-generation")
prompt = "The Gion neighborhood in kyoto is famous for"
outputs = llm(prompt, max_length=100)
print(outputs[0]["generated_text"])
```

#### Text Generation
here, a long paragraph is summarized and given by the LLM based on the `max_length`
```python
llm = pipeline("summarization", model="facebook/bart-large-cnn")\
long_text = """askjgfd;kdcosdhcskdmlsdhfousnc"""
outputs = llm(long_text, max_length=60, clean_up_tokenization_spaces=True)
print(outputs[0]["summary_text"])
```

#### Question Answering
here, a question from a given context is asked to the model and the LLM answers it.
```python
llm = pipeline("question-answering")
context = "Walking amid Gion's Machiya wooden houses was a mesmerizing experience"
question = "What are Machiya houses made of?"
outputs = llm(question=question, context=context)
print(outputs["answer"])
```

#### Language Translation
here, a sentence from one language is translated to another
```python
llm = pipeline("translation_en_to_es", model="Helsiniki-NLP/opus-mt-en-es")
text = "Walking amid Gion's Machiya wooden houses was a mesmerizing experience"
outputs = llm(text, clean_up_tokenization_spaces=True)
print(outputs[0]["translation_text"])
```

The Transformers architecture is for processing, understanding and generating text in human language.
- they do not rely on RNNs
- captures long - range dependencies in text with **attention mechanisms** and **Positional Encoding**
- tokens are handled simultaneously instead of sequentially so much faster.

Original Transformer consists of 2 main components - encoder and decoder
in each layer, attention mechanism is applied to capture semantic meanings
![[Pasted image 20240311231244.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240311231244.png/?raw=true)
`PyTorch` Transformers can be implemented as follows
`d_model` : this is the embedding dimension used to represent inputs, intermediate data and outputs
`n_heads`: this refers to the number of attention heads, which perform parallel computations, usually set as a divisor of the number of embeddings
`num_encoder_layers` - number of encoder layers
`num_decoder_layers` - number of decoder layers

the three types of transformers are:
1. **Encoder- decoder** type, where understanding and reciprocating is required - T5, BART - language translation, text summarization
2. **Encoder only** requires only understanding and not generating - BERT - used for text classification and Q&A
3. **Decoder only** requires only reciprocating based on given prompts - GPT - Text Generation

```python
d_model = 512
n_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
model = nn.Transformer(d_model=d_model,
    nhead=n_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers
)
print(model)
```

# The Transformers Architecture
the transformers architecture uses attention mechanisms to process words/tokens. this makes the process more faster than RNNs as they process tokens sequentially.
![[Pasted image 20240314170206.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240314170206.png/?raw=true)
here, all tokens are weighed with importance weights simultaneously.
but to weigh this, attention mechanisms require info about the position of the tokens. a position encoder precedes the attention mechanism
given an embedding vector of a sequence, a vector is created with values describing the positions of each token, and are made unique by sine and cosine functions. then the embeddings are added with the position vector to get the positional encodings

```python
class PositionalEncoder(nn.Module):
	def __init__(self, emb_d, max_seq_length=512):
		super(PositionalEncoder, self).__init__()
		self.emb = emb_d
		self.max_seq_length = max_seq_length

		pe = torch.zeros(max_seq_length)
		position = torch.arange(0, max_seq_length,
		 dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, emb_d, 2, dtype=torch.float) * -(math.log(10000.0) / emb_d))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer("pe", pe)
	
	def forward(self, x):
		x = x + self.pe[:, :x.size(1)]
		return x
```
what we are doing here is, we get a pe matrix, with elements up to the `max_length`. then we calculate the `div_term` for each position, which is used to generate the unique frequency to pass to sin and cos for a unique value for each position. a unique value is generated for each embedding, so `div_term` is has `emb_d` components and each alternative embedding gets a unique positional encoding with either sin or cos.

self attention mechanisms find the words with utmost importance and use them to process sequences. to find this, first the given embedding tokens are divided into 3 vectors using linear transforms with trained weights: query, key and value
then scaled dot product is applied to the query and key matrices to yield a matrix of attention scores for each word. this is then passed to a softmax function to get the relevance scaling for the attention scores. so each word has a score between 0 and 1 and all scores add up to 1. so for a given query - we have attention scores for other words.
![[Pasted image 20240314171912.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240314171912.png/?raw=true)
attention scores are multiplied with the values matrix to get updated token embeddings.
![[Pasted image 20240314172033.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240314172033.png/?raw=true)
the above process is done by one attention head. in reality, there are multiple heads working in parallel handling multiple things like context, sentiment, etc.
![[Pasted image 20240314172203.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240314172203.png/?raw=true)

```python
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
	def __init__(self, emb_d, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.emb_d = emb_d
		self.head_dim = emb_d // num_heads

		self.query_linear = nn.Linear(emb_d, emb_d)
		self.key_linear = nn.Linear(emb_d, emb_d)
		self.value_linear = nn.Linear(emb_d, emb_d)
		self.output_linear = nn.Linear(emb_d, emb_d)
	def split_heads(self, x, batch_size):
		x = x.view(batch_size, -1, self.num_heads, self.head_dim)
		return x.permute(0, 2, 1, 3).contiguous().
			view(batch_size * self.num_heads, -1, self.head_dim)

	def compute_attention(self, query, key, mask=None):
		scores = torch.matmul(query, key.permute(1, 2, 0))
		if mask is not None:
			scores = scores.masked_fill(mask == 0, float("1e-9"))
		attention_weights = F.softmax(scores, dim=-1)
		return attention_weights

	def forward(self, query, key, value, mask=None):
		batch_size = query.size(0)
		query = self.split_heads(self.query_linear(query), batch_size)
		key = self.split_heads(self.key_linear(key), batch_size)
		value = self.split_heads(self.value_linear(value), batch_size)

		attention_weights = self.compute_attention(query, key, mask)

		output = torch.matmul(attention_weights, value)
		output = output.view(batch_size, self.num_heads, -1, self.head_dim).
			permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.emb_d)
		return self.output_linear(output)
```
the mask here is instead of padding in regular NLP tasks. we can't use padding for attention mechanisms so we use masking instead.
![[Pasted image 20240321010212.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240321010212.png/?raw=true)
now for a transformer that only needs to understand the text and give simple outputs we need an encoder. to implement it,
![[Pasted image 20240321004626.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240321004626.png/?raw=true)
```python
class FeedForwardSubLayer(nn.Module):
	def __init__(self, d_model, d_ff):
		super(FeedForwardSubLayer, self).__init__()
		self.fc1 = nn.Linear(d_model, d_ff)
		self.fc2 = nn.Linear(dff, d_model)
		self.relu = nn.ReLU

	def forward(self, x):
		return self.fc2(self.relu(self.fc1(x)))
```
this is for capturing more complex patterns from the attention mechanisms. so the attention mechanism and feed forward constitute one layer in the encoder. to create that
```python
class EncoderLayer(nn.module):
	def __init__(self, d_model, num_heads, d_ff, dropout):
		self.self_attn = MultiHeadAttention(d_model, num_heads)
		self.feed_forward = FeedForwardSubLayer(d_model, d_ff)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
	def forward(self, x, mask):
		attn_output = self.self_attn(x, x, x, mask)   #attentions
		x = self.norm1(x + self.dropout(attn_output)) #normalization + dropout
		ff_output = self.feed_forward(x)  #feed forward for capturing more
		x = self.norm2(x + self.dropout(ff_output)) #normalization + dropout
		return x
```
lets bring multiple encoders to get out transformers architecture.
```python
class TransformerEncoder(nn.Module):
	def __init__(self, vocab_size, d_model, num_layers, num_heads,
	 d_ff, dropout, max_sequence_length):
		 super(TransformerEncoder, self).__init__()
		self.embedding = nn.Embedding(vocab_size, d_model)
		self.positional_encoding = PositionalEncoding(d_model,
		max_sequence_length)
		self.layers = nn.ModuleList(
		[EncoderLayer(d_model, num_heads, d_ff, dropout)
		for _ in range(num_layers)]
		)
	def forward(self, x):
		x = self.embedding(x)  # getting the embeddings
		x = self.positional_encoding(x)    #finding the positional encodings
		for layer in self.layers:
			x = layer(x, mask)     #passing through each layer
		return x
```
the above is the transformer body. for the transformer head, we define either a classifier or a regressor
```python
class ClassifierHead(nn.Module):
	def __init__(self, d_model, num_classes):
		super(ClassifierHead, self).__init__()
		self.fc = nn.Linear(d_model, num_classes)
	def forward(self, x):
		logits = self.fc(x)
		return F.log_softmax(logits, dim=-1)
```

Decoder only transformers involves two differences from encoder only. they have something called masked attention mechanisms, which helps in processing sequences iteratively. and the other thing is a softmax layer at end over the whole vocab to generate probabilities for each word and predict.
By passing this matrix to the attention heads, each token in the sequence only pays attention to "past" information on its left-hand side.
![[Pasted image 20240321185704.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240321185704.png/?raw=true)
we just feed the decoder with the masked sequence when evaluating.

```python
class DecoderOnlyTransformer(nn.Module):
	def __init__(self, vocab_size, d_model, num_layer, num_heads,
	 d_ff, dropout, max_sequence_length):
		 super(DecoderOnlyTransformer, self)._init__()
		 self.embedding = nn.Embedding(vocab_size, d_model)
		 self.positional_encoding = PositionalEncoding(d_model, 
		 max_sequence_length)
		 self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, 
		 d_ff, dropout) for _ in range(num_layers)])
		 self.fc = nn.Linear(d_model, vocab_size)

	def forward(self, x, self_mask):
		x = self.embedding(x)
		x = self.positional_encoding(x)
		for layer in self.layers:
			x = layer(x, mask)
		x = self.fc(x)
		return F.log_softmax(x, dim=-1)
```

for encoder and decoder both, we have something called cross attention mechanism, which connects both the encoder and the decoder. this is added to each decoder layer and takes in the input of the final encoder layer and the info processed by the decoder. this helps look back at the input sequence that went through the encoder.
![[Pasted image 20240321192943.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240321192943.png/?raw=true)

inside the decoder class we just add the cross attention layer
![[Pasted image 20240321193117.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240321193117.png/?raw=true)
here y is the encoder's final layer's output. and finally we have the same linear and softmax layers.
![[Pasted image 20240321193259.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240321193259.png/?raw=true)
when training, the output embedding is the labels, i.e. the actual things we want the model to generate. if its translation, then output embedding should be the translated sentences. when evaluating, the model gets an empty output embedding and proceeds to generate one on the fly layer by layer.
![[Pasted image 20240321193519.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240321193519.png/?raw=true)

# Pretrained LLMs
to utilize pretrained LLMs we usually don't use the pipeline module. instead, we use the Auto class for importing models, which gives us more flexibility.
```python
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

class SimpleClassifier(nn.Module):
	def __init__(self, input_size, num_classes):
		super(SimpleClassifier, self).__init__()
		self.fc = nn.Linear(input_size, num_classes)
	def forward(self, x):
		return self.fc(x)

inputs = tokenizer(
				   text, return_tensors="pt", padding=True,
				    truncation=True, max_length=64)
outputs = model(**inputs)
pooled_output = outputs.pooler_output  #aggregates the output

classifier_head = SimpleClassifier(pooled_output.size(-1), num_classes=2)
logits = classifier_head(pooled_output)
probs = torch.softmax(logits, dim=1)
```
some classes exist for specific tasks like text classification, to use a pre configured model.
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification(model_name)
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

predicted_class = torch.argmax(logits, dim=1).item()
```
same is available for other tasks too.
for text generation, the LLM predicts the next word for a given input sequence.
input - segment of the whole sentence. this gets shifted one by one, word by word and it goes on predicting the next word.
for text summarization and translation, we use `AutoModelForSeq2SeqLM`
for question answering, there are 3 types:
1. Extractive QA: here, the answer for the question is extracted from the context given separately. an Encoder architecture is used.
2. Open Generative QA: the answer is not extracted from the context, but it inferences from the context and gives the answer. it requires Decoder architecture
3. Closed Generative QA: here, no context is provided and the model generates an answer based on its own knowledge. this requires encoder decoder architecture.
for extractive QA, supervised learning takes place, where each token in the context is given a probability of it being the start or finish word. then we extract those and the words in between to get the answer.`AutoModelForQuestionAnswering` is used.

## Transfer Learning
transfer learning is the process of using a model, which was already trained with one specific task, to do another. one way of transfer learning is to fine tune, which is to train them model on domain specific data. to do that, 
```python
from transformers import Trainer, TrainingArguments

training_args = TrainerArguments(
								 output_dir="./smaller_bert_finetuned",
								 per_device_train_batch_size=8,
								 num_train_epochs=3,
								 evaluation_strategy="steps",
								 eval_steps=500,
								 save_steps=500,
								 logging_dir="./log"
)
trainer = Trainer(
				  model=model, 
				  args=training_args,
				  train_dataset=tokenized_dataset["train"],
				  eval_dataset=tokenized_dataset["test"]
)

trainer.train()
```
`tokenized_dataset` is a `DatasetDict` object which belongs to transformers.
for inference,
```python
example_input = tokenizer("I am absolutely.. ..")
output = model(**example_input)
predicted_label = torch.argmax(output.logits, dim=1).item()
```


# Evaluating LLMs
Using basic accuracy for sentiment classification
```python
from transformers import pipeline
from sklearn.metrics import accuracy_score
sentiment_analysis = pipeline("sentiment-analysis")
text_examples = [
				 {"text":"I love this product!", "label":1},
				 {"text":"I love this product!", "label":1},
				 {"text":"I love this product!", "label":1},
				 {"text":"I love this product!", "label":1},
]
predictions = sentiment_analysis([example["text"] for example in test_examples])

true_labels = [example["label"] for example in text_examples]
predicted_labels = [1 if pred["label"] == "POSITIVE" else 0 for pred in predictions]
accuracy = accuracy_score(true_labels, predicted_labels)
print(accuracy)
```

The evaluate library from transformers is used to calculate LLM based metrics
```python
import evaluate
accuracy = evaluate.load("accuracy")
print(accuracy.description) #prints what the metric is
print(accuracy.features) #prints what are the parameters required

print(accuracy.compute(references=targets,  predictions=predicted_labels))
#prints the calculated metric
```

different metrics are used for different tasks
![[Pasted image 20240312191218.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240312191218.png/?raw=true)

some metrics are
### Perplexity
the model's ability to be able to predict the next word. this is used for text generation. takes in output logits distribution to give out the metric

### ROUGE score
used in text summarization. quantifies similarity between summaries predicted and reference summaries.

### BLEU score
used in translation to measure translation quality by the correspondence between the LLM's prediction and references given.

### METEOR
This is also used in translation, but overcomes limitations of BLEU and ROUGE by using more linguistic aspects. but its more computationally expensive.

### Exact match
this is used for QA. basic accuracy and it gets better with F1 score since without it, the metric is sensitive

## Finetuning with human feedback
