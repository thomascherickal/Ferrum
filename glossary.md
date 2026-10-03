# Glossary of LLM and SLM Engineering

A plain-English guide to the words you will meet when you build, train, shrink,
and run language models. It is written for beginners in Generative AI. No
maths background is assumed beyond school algebra, and every term is explained
using only terms that come before it, or that it points you to.

Terms are grouped by topic, in roughly the order you would learn them. If a
definition mentions a word you don't know yet, look for it in *italics* and
search this page for it.

Where a term maps directly onto something in this repository, a short
**In Ferrum** note says where to find it.

---

## Contents

1. [The big picture](#1-the-big-picture)
2. [Maths building blocks](#2-maths-building-blocks)
3. [Turning text into numbers](#3-turning-text-into-numbers)
4. [Inside a Transformer](#4-inside-a-transformer)
5. [Training a model](#5-training-a-model)
6. [Fine-tuning and alignment](#6-fine-tuning-and-alignment)
7. [Generating text (inference)](#7-generating-text-inference)
8. [Making models small and fast](#8-making-models-small-and-fast)
9. [Files, formats, and tools](#9-files-formats-and-tools)
10. [Building applications with LLMs](#10-building-applications-with-llms)
11. [Evaluating models](#11-evaluating-models)
12. [Rules of thumb](#12-rules-of-thumb)

---

## 1. The big picture

### Artificial Intelligence (AI)
The broad field of making computers do tasks that normally need human
intelligence, such as understanding language, recognising images, or making
decisions.

### Machine Learning (ML)
A part of AI where, instead of writing rules by hand, you show the computer
many examples and let it *learn* the rules. A spam filter that learns from
thousands of labelled emails is machine learning.

### Deep Learning
Machine learning that uses *neural networks* with many layers stacked on top of
each other ("deep" refers to the number of layers). Almost all modern language
models are deep learning models.

### Neural Network
A program made of many simple maths units ("neurons") connected in layers. Each
connection has a number called a *weight*. Learning means adjusting those
weights until the network's outputs are good. Despite the name, it is only
loosely inspired by the brain.

### Generative AI (Gen AI)
AI that *creates* new content (text, images, audio, code) rather than only
classifying or predicting a label. ChatGPT, Claude, and image generators are
generative AI.

### Language Model (LM)
A model that has learned the patterns of a language well enough to predict what
text comes next. Given "The cat sat on the", it assigns high probability to
"mat" and low probability to "refrigerator".

### Large Language Model (LLM)
A language model with a very large number of *parameters*, usually billions to
hundreds of billions, trained on a huge amount of text. Size gives broad
knowledge and flexible reasoning, but costs a lot of memory and compute to run.
Examples: GPT-4, Claude, Llama 3 70B.

### Small Language Model (SLM)
A language model small enough to train or run on modest hardware such as a
laptop CPU, a phone, or a Raspberry Pi. There is no official cutoff; the term
usually covers models from a few million up to a few billion parameters.
SLMs trade breadth of knowledge for speed, low cost, privacy (data stays on
the device), and the ability to specialise well on a narrow task.
**In Ferrum:** the whole project is an SLM engine; see `ReadMe.md`.

### Foundation Model
A large model trained on broad data that serves as a starting point for many
different tasks. You usually *fine-tune* or *prompt* a foundation model rather
than training one yourself.

### Base Model (Pre-trained Model)
A model straight out of *pre-training*. It is good at continuing text but does
not reliably follow instructions or hold a conversation. Ask it a question and
it may reply with more questions.

### Instruct Model / Chat Model
A base model that has been further trained (*instruction tuning*, *RLHF*) to
follow instructions and chat helpfully. Model names often end in `-Instruct`
or `-Chat`.

### Open-weight Model vs Closed Model
An **open-weight** model publishes its *weights* so anyone can download and run
it (Llama, Qwen, Mistral, Gemma, Phi). A **closed** model is only reachable
through the company's API (GPT-4, Claude). "Open source" strictly also means
the training code and data are open, which is rarer.

### Parameter
One learned number inside a model. A "7B model" has about 7 billion
parameters. Parameters are mostly *weights* and *biases*. More parameters
means more capacity to learn, and more memory to store.

### Weight
A parameter that multiplies an input. Training adjusts weights so the model's
predictions improve. People often say "the weights" to mean the whole trained
model file.

### Bias (parameter)
A parameter that is *added* after a multiplication, letting a neuron shift its
output up or down. Not the same as *bias* in the sense of unfairness, which
is also an important topic in AI.

### Training
The process of adjusting a model's parameters using data so it gets better at
its task. Expensive: it needs many passes over a lot of data.

### Inference
Using a trained model to produce outputs, for example generating a reply to a
prompt. No learning happens. Inference is what users experience, so its speed
and cost matter most in production.

### Multimodal Model
A model that handles more than one kind of data, such as text plus images or
audio.

---

## 2. Maths building blocks

### Scalar, Vector, Matrix
A **scalar** is a single number. A **vector** is a list of numbers, like
`[0.2, -1.3, 0.7]`. A **matrix** is a grid of numbers with rows and columns.
Neural networks are mostly vectors and matrices being multiplied together.

### Tensor
The general name for a block of numbers with any number of dimensions. A
vector is a 1-D tensor, a matrix is 2-D, and a batch of matrices is 3-D.
Deep learning libraries are built around tensors.
**In Ferrum:** `ferrum_core/src/tensor.rs`.

### Shape
The size of a tensor along each dimension. A tensor of shape `[32, 128, 512]`
might mean 32 sequences, 128 tokens each, 512 numbers per token.

### Dot Product
Multiply two vectors element by element and add the results. It measures how
much two vectors "point the same way", which is the core of *attention*.

### Matrix Multiplication (matmul, GEMM)
Combining two matrices by taking dot products of rows and columns. It is by far
the most common and most expensive operation in a language model. **GEMM**
("General Matrix Multiply") is the standard name for an optimised matmul
routine.

### Hidden Size (Model Dimension, d_model)
How many numbers represent each token inside the model. A model with hidden
size 4096 describes every token with a vector of 4096 numbers. Bigger means
more expressive and more expensive.

### Activation Function
A simple non-linear function applied after a matrix multiplication. Without
it, stacking layers would collapse into one big linear function and the
network could not learn complex patterns. Common ones:
- **ReLU**: keeps positive numbers, turns negatives into 0.
- **GELU**: a smooth version of ReLU, used in GPT-style models.
- **SiLU / Swish**: another smooth curve, used in Llama-style models.
- **SwiGLU**: a "gated" combination using SiLU, used in the feed-forward
  layers of Llama, Qwen, and Mistral.

**In Ferrum:** `ferrum_core/src/activation.rs`.

### Logits
The raw scores the model produces for each possible next token, before they
are turned into probabilities. Higher logit means "more likely".

### Softmax
A function that turns a list of logits into probabilities that are all
positive and add up to 1. It exaggerates differences: the biggest logit gets
most of the probability.

### Probability Distribution
A list of possible outcomes with a probability for each, adding up to 1. A
language model outputs a probability distribution over its whole *vocabulary*
at every step.

### Normalization (LayerNorm, RMSNorm)
Rescaling a vector so its values stay in a sensible range. This keeps training
stable. **LayerNorm** subtracts the mean and divides by the standard
deviation. **RMSNorm** skips the mean and only divides by the root-mean-square;
it is cheaper and used by Llama and Qwen.

### Floating Point (FP32, FP16, BF16)
How computers store decimal numbers.
- **FP32** (32-bit, 4 bytes): standard precision, used for training on CPUs.
- **FP16** (16-bit, 2 bytes): half the memory, smaller range of values.
- **BF16** (bfloat16, 2 bytes): same range as FP32 but less precision; popular
  for GPU training because it rarely overflows.

### Precision
How many bits each number uses. Lower precision saves memory and speeds up
computation but loses some accuracy. See *quantization*.

---

## 3. Turning text into numbers

### Token
The basic unit of text a model reads and writes. A token can be a whole word
("cat"), part of a word ("un", "believ", "able"), a punctuation mark, or a
single byte. In English, one token is roughly ¾ of a word on average.

### Tokenizer
The component that splits text into tokens and converts them to numbers
(*token IDs*), and converts IDs back into text. A model only works with the
tokenizer it was trained with.
**In Ferrum:** `ferrum_core/src/tokenizer.rs` and, for imported models,
`gguf_tokenizer.rs`.

### Tokenization / Detokenization
**Tokenization** is text → token IDs. **Detokenization** (decoding) is token
IDs → text.

### Token ID
The integer that stands for a token in the vocabulary, for example
`"cat" → 9246`. The model never sees letters, only these numbers.

### Vocabulary (Vocab)
The fixed list of all tokens a tokenizer knows. Typical sizes are 32,000 to
150,000 tokens. Larger vocabularies make text shorter (fewer tokens) but make
the *embedding* and *LM head* layers bigger, which matters a lot in small
models.

### Byte Pair Encoding (BPE)
The most common way to build a vocabulary. Start with single characters (or
bytes), then repeatedly merge the most frequent neighbouring pair into a new
token, until the vocabulary reaches the target size. Frequent words end up as
one token; rare words are split into pieces.

### Byte-level BPE
BPE that starts from the 256 possible bytes instead of characters. Because any
text is made of bytes, it can encode *anything* (emoji, any language, code)
without an "unknown" token.
**In Ferrum:** the built-in tokenizer is byte-level BPE.

### WordPiece, SentencePiece, Unigram
Other tokenization methods. **WordPiece** (used by BERT) is similar to BPE.
**SentencePiece** is a library that treats spaces as ordinary symbols and
supports both BPE and **Unigram**, a method that starts with a big vocabulary
and prunes it down.

### Special Tokens
Reserved tokens with a job rather than a meaning:
- **BOS** (beginning of sequence) marks the start of text.
- **EOS** (end of sequence) tells the model, or the program, to stop.
- **PAD** fills shorter sequences so a batch has equal lengths.
- **UNK** (unknown) stands in for text the tokenizer cannot encode. Byte-level
  tokenizers don't need it.

### Context Window (Context Length)
The maximum number of tokens a model can look at in one go, covering both the
prompt and the generated reply. Ranges from a few hundred tokens in tiny SLMs
to over a million in some LLMs. Text beyond the window is simply not seen.

### Chat Template (Chat Format)
The exact text layout a chat model expects, with special markers for who is
speaking, for example
`<|user|>Hello<|end|><|assistant|>`. Using the wrong template makes a chat
model behave badly even though the weights are fine.
**In Ferrum:** `ferrum_core/src/chat_format.rs`.

---

## 4. Inside a Transformer

### Transformer
The neural network architecture behind nearly all modern language models,
introduced in the 2017 paper "Attention Is All You Need". Its key idea is
*attention*, which lets every token look directly at every other relevant
token instead of reading strictly left to right.

### Encoder, Decoder, Encoder-Decoder
The original Transformer had two halves. An **encoder** reads the whole input
at once to understand it (BERT is encoder-only; good for classification and
search). A **decoder** generates output one token at a time. **Encoder-decoder**
models (T5) use both, which suits translation.

### Decoder-only Model
A Transformer with only the decoder half. GPT, Llama, Qwen, Mistral, Claude,
and almost every chat LLM and SLM are decoder-only.

### Causal Language Modelling (CLM)
Training a model to predict the next token using only the tokens *before* it.
"Causal" means the model can't peek at the future. This is how decoder-only
models are trained.

### Embedding
A vector of numbers that represents a token (or a word, sentence, or image) so
that similar meanings end up as nearby vectors. The **embedding layer** is a
big lookup table: token ID in, vector out.

### Attention
A mechanism that lets each token gather information from other tokens,
weighting each by how relevant it is. In "The animal didn't cross the street
because **it** was tired", attention helps "it" focus on "animal".

### Self-Attention
Attention where a sequence attends to itself: every token in a sentence looks
at the other tokens in the same sentence.

### Query, Key, Value (Q, K, V)
The three vectors each token produces for attention. Think of a library:
- The **query** is what this token is looking for.
- The **key** is the label on each other token, saying what it offers.
- The **value** is the actual content that gets passed along.

A token compares its query against every key (with a *dot product*), turns the
scores into weights with *softmax*, and takes the weighted average of the
values.

### Attention Head and Multi-Head Attention
An **attention head** is one independent copy of the attention calculation.
**Multi-head attention** runs several heads side by side, so one head can track
grammar, another can track which noun a pronoun refers to, and so on. Their
outputs are combined.

### Causal Mask
A rule applied in attention that blocks each token from seeing tokens after
it. This keeps training honest: the model can't cheat by looking at the answer.

### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
Ways to save memory during inference. In normal attention, every head has its
own keys and values. **MQA** makes all heads share one set. **GQA** is the
middle ground: groups of query heads share a key/value head. GQA is standard in
Llama 3, Qwen, and Mistral because it shrinks the *KV cache* with little loss in
quality.

### Positional Encoding
Attention on its own has no sense of word order: "dog bites man" and "man bites
dog" would look the same. Positional encodings add information about each
token's position.

### RoPE (Rotary Position Embedding)
The most common modern positional encoding. It rotates each query and key
vector by an angle that depends on the token's position, so the attention score
between two tokens depends on how far apart they are. Used by Llama, Qwen,
Mistral, and most current models.
**In Ferrum:** `apply_rope` in `ferrum_core/src/llm.rs`.

### Feed-Forward Network (FFN, MLP)
The second half of each Transformer block: two or three matrix multiplications
with an *activation function* between them, applied to each token separately.
Attention moves information *between* tokens; the FFN processes information
*within* each token. Most of a model's parameters live here.

### Residual Connection (Skip Connection)
Adding a layer's input back onto its output: `output = input + layer(input)`.
This gives information and gradients an easy path through deep networks and is
essential for training models with many layers.

### Transformer Block (Layer)
One repeating unit of a Transformer: normalization → attention → residual add →
normalization → feed-forward → residual add. Models stack many identical
blocks: a small SLM might have 4–12, a large LLM 80 or more.
**In Ferrum:** `TransformerBlock` in `ferrum_core/src/layer.rs` and
`LlamaBlock` in `llm.rs`.

### Pre-Norm vs Post-Norm
Where the normalization goes inside each block. **Pre-norm** (normalize
*before* attention and FFN) trains more stably and is what modern models use.

### LM Head (Unembedding, Output Layer)
The final layer that turns the model's last hidden vector into one logit per
vocabulary token, giving the scores for the next token.

### Weight Tying
Using the same matrix for the *embedding layer* and the *LM head*. This saves a
large share of parameters in small models, where the vocabulary matrix can be a
big fraction of the whole model.

### Mixture of Experts (MoE)
An architecture where each block has many feed-forward "experts" but each token
only uses a few of them, chosen by a small *router* network. The model has many
parameters in total but only uses a fraction per token, so it is cheaper to run
than a dense model of the same size. Mixtral and DeepSeek-V3 are MoE models.

### Dense Model
The opposite of MoE: every parameter is used for every token.

### FlashAttention
A faster way of computing exactly the same attention result, by arranging the
calculation to use the GPU's small fast memory efficiently. It is an
implementation trick, not a different model.

### Sliding Window Attention
Each token attends only to the last *N* tokens rather than the whole context.
This reduces memory and compute for long texts.

---

## 5. Training a model

### Dataset and Corpus
A **dataset** is the collection of examples used for training. For language
models, it is usually a **corpus**: a large body of text such as books, web
pages, or code.
**In Ferrum:** `ferrum_core/src/dataset.rs` cleans and checks a corpus before
training.

### Data Cleaning and Deduplication
Removing junk (broken encodings, boilerplate, spam) and repeated copies of the
same text from a corpus. Clean data matters more than almost any other choice,
especially for small models, which have little capacity to waste on noise.

### Pre-training
The first and most expensive training stage: teaching a model general language
by having it predict the next token across a huge corpus.

### Next-Token Prediction
The training task for decoder-only models. For every position in a text, the
model guesses the next token and is scored on how much probability it gave the
right one. This single simple task, at enough scale, produces surprisingly
general abilities.

### Sequence (Training Window) and Stride
Training text is cut into fixed-length **sequences** (windows), for example 256
tokens each. The **stride** is how far you move along the text before starting
the next window. A stride equal to the window length gives non-overlapping
windows; a smaller stride makes them overlap.

### Batch and Batch Size
A **batch** is a group of training sequences processed together before
updating the weights. **Batch size** is how many sequences are in it. Larger
batches give smoother learning but need more memory.

### Step (Iteration)
One update of the model's weights, after processing one batch.

### Epoch
One complete pass through the whole training dataset. Large models often train
for about one epoch; small models on small datasets may train for many.

### Token Budget
The total number of tokens you plan to train on. It is a more useful measure of
training size than epochs when datasets differ in size.

### Loss and Loss Function
The **loss** is a single number saying how wrong the model's predictions are;
lower is better. The **loss function** is the formula that computes it.
Training is the search for weights that make the loss small.
**In Ferrum:** `ferrum_core/src/loss.rs`.

### Cross-Entropy Loss
The standard loss for language models. It is large when the model gave the
correct next token a low probability and small when it gave it a high one.

### Perplexity
A friendlier way to read cross-entropy: `perplexity = e^loss`. It roughly means
"how many tokens the model is torn between, on average". A perplexity of 10
means the model is about as uncertain as picking from 10 equally likely
options. Lower is better.

### Gradient
For each parameter, the direction and amount by which a small change would
increase the loss. Training moves each parameter a little in the *opposite*
direction to reduce the loss.

### Backpropagation (Backprop)
The algorithm that computes the gradient for every parameter efficiently by
working backwards from the loss through each layer, using the chain rule from
calculus.

### Gradient Checking
Testing that hand-written backpropagation is correct by comparing its gradients
with ones estimated numerically (nudge a weight, see how the loss changes).
**In Ferrum:** the backward passes are gradient-checked in the test suite.

### Gradient Descent
The basic learning rule: repeatedly compute the gradient and take a small step
downhill. **Stochastic gradient descent (SGD)** does this using a random batch
each time instead of the whole dataset.

### Optimizer
The algorithm that decides how to update weights from their gradients. SGD is
the simplest; Adam and AdamW are the usual choice for Transformers.
**In Ferrum:** `ferrum_core/src/optim.rs`.

### Adam and AdamW
**Adam** keeps a running average of each parameter's past gradients and their
size, and uses them to give each parameter its own adaptive step size.
**AdamW** is Adam with *weight decay* applied correctly; it is the default
optimizer for training language models.

### Weight Decay
A small pull of every weight towards zero on each step. It discourages the
model from relying on huge weights and helps prevent *overfitting*.

### Learning Rate (LR)
How big a step the optimizer takes on each update. Too high and training
becomes unstable or blows up; too low and it learns very slowly. It is the most
important *hyperparameter* to get right.

### Learning Rate Schedule, Warmup, and Cosine Decay
A **schedule** changes the learning rate during training. **Warmup** starts
with a tiny learning rate and ramps it up over the first steps, which avoids
early instability. **Cosine decay** then lowers it smoothly along a cosine curve
towards a small final value, so the model settles into a good solution.

### Gradient Clipping
If the gradients become too large, scale them down to a maximum size before
updating. This stops a single bad batch from wrecking the model.

### Exploding and Vanishing Gradients
Problems in deep networks where gradients grow huge (**exploding**, making
training diverge) or shrink to almost nothing (**vanishing**, making early
layers stop learning). Residual connections, normalization, and gradient
clipping all help.

### Gradient Accumulation
Adding up gradients from several small batches before doing one update. It
simulates a large batch when memory only fits a small one.

### Mixed-Precision Training
Doing most calculations in 16-bit numbers for speed while keeping a 32-bit
master copy of the weights for accuracy. Common on GPUs.

### Hyperparameter
A setting you choose *before* training, rather than one the model learns:
learning rate, batch size, number of layers, hidden size, and so on.

### Overfitting and Underfitting
**Overfitting**: the model memorises the training data and does worse on new
data. Training loss keeps falling while validation loss rises. **Underfitting**:
the model is too small or trained too little to learn the patterns at all.

### Train / Validation / Test Split
Dividing data into three parts. The **training set** is used to learn. The
**validation set** is checked during training to tune settings and spot
overfitting. The **test set** is kept untouched until the end for a final,
honest score.

### Validation Loss
The loss measured on the validation set. It is the best everyday signal of
whether the model is genuinely improving.

### Early Stopping
Stopping training automatically when validation loss stops improving for a
while, and keeping the best version seen so far.

### Checkpoint and Resume
A **checkpoint** is a saved snapshot of the model (and usually the optimizer
state) during training. **Resuming** continues training from a checkpoint after
a crash or pause, without starting over.

### Random Seed and Determinism
A **seed** is the starting number for the random number generator. Using the
same seed makes a run repeatable. A **deterministic** system gives identical
results every time for the same inputs.
**In Ferrum:** results are bit-for-bit identical at any thread count;
`ferrum_core/src/rng.rs`.

### Scaling Laws
Measured relationships showing that loss falls predictably as you increase
parameters, data, and compute. They let teams predict how good a bigger model
will be before training it.

### Chinchilla-Optimal
A finding from DeepMind's 2022 "Chinchilla" paper: for the best model at a
fixed compute budget, train on about **20 tokens per parameter**. Many modern
SLMs deliberately train far beyond this (hundreds or thousands of tokens per
parameter) because a smaller, longer-trained model is cheaper to *run*.

### FLOPs and Compute
**FLOPs** (floating-point operations) count the arithmetic a model performs.
"Compute" means the total amount of calculation used. A common estimate for
training cost is `6 × parameters × training tokens` FLOPs.

### Synthetic Data
Training data generated by another model rather than written by people. Widely
used to train SLMs (for example, Microsoft's Phi models) on clean, textbook-like
material.

### Distributed Training
Spreading training over many processors. **Data parallelism** gives each device
a copy of the model and different data; **model (tensor or pipeline)
parallelism** splits the model itself across devices when it is too big for
one.

---

## 6. Fine-tuning and alignment

### Transfer Learning
Reusing what a model learned on one task as a starting point for another. It is
why you rarely need to train from scratch.

### Fine-tuning
Continuing to train an already-trained model on a smaller, focused dataset so
it gets better at a specific task, style, or domain.
**In Ferrum:** `finetune-gguf` in `slm_cli`, backed by `llm_train.rs`.

### Full Fine-tuning
Fine-tuning that updates *every* parameter. Best quality, but needs memory for
the weights, gradients, and optimizer state, which is several times the model's
size.

### Continued Pre-training (Domain Adaptation)
Further next-token training on raw text from a specific field (medical, legal,
your company's documents) so the model absorbs its vocabulary and facts.

### Supervised Fine-tuning (SFT) and Instruction Tuning
Training on example pairs of *(instruction, ideal response)*. This is how a
base model becomes an *instruct* or *chat* model.

### Parameter-Efficient Fine-tuning (PEFT)
Fine-tuning methods that train only a small number of extra or selected
parameters while freezing the rest. Much cheaper in memory and storage.

### LoRA (Low-Rank Adaptation)
The most popular PEFT method. Instead of changing a big weight matrix directly,
LoRA learns two small matrices whose product is added to it. You might train
under 1% of the parameters and save the result as a small "adapter" file.

### QLoRA
LoRA applied to a model whose frozen base weights are stored *quantized* (in
4-bit). It lets you fine-tune fairly large models on a single consumer GPU.

### Adapter
A small add-on set of trained weights (such as a LoRA) that changes a base
model's behaviour. Several adapters can be swapped onto one base model.

### Catastrophic Forgetting
When fine-tuning on a new task makes a model lose skills it had before. Mixing
in some general data and using small learning rates both help.

### Alignment
Making a model's behaviour match human intentions and values: helpful, honest,
and harmless.

### RLHF (Reinforcement Learning from Human Feedback)
An alignment method. People rank pairs of model responses; a **reward model**
learns to predict those preferences; then the language model is trained with
reinforcement learning to produce responses the reward model scores highly.

### Reward Model
A model that scores how good a response is, trained on human (or AI)
preference data. Used in RLHF.

### DPO (Direct Preference Optimization)
A simpler alternative to RLHF that trains directly on pairs of "preferred" and
"rejected" responses, with no separate reward model or reinforcement learning
loop.

### Knowledge Distillation
Training a small **student** model to imitate a larger **teacher** model,
often by matching the teacher's full probability distribution rather than just
the correct answer. A key way to build strong SLMs.

---

## 7. Generating text (inference)

### Prompt and Completion
The **prompt** is the text you give the model. The **completion** (or
response) is what it generates.

### System Prompt
Instructions placed before the conversation that set the model's role, rules,
and tone, for example "You are a concise assistant for a bank's support team."

### Autoregressive Generation
Generating one token at a time, adding each new token to the input, and
running the model again to get the next one. This is why replies appear word
by word.

### Prefill and Decode
The two phases of generation. **Prefill** processes the whole prompt in one
parallel pass; it is limited by raw compute. **Decode** then produces the reply
one token at a time; it is limited by how fast weights can be read from memory.

### KV Cache
During decode, the keys and values for earlier tokens never change, so they are
stored instead of recomputed. This makes generation much faster but uses memory
that grows with the context length.
**In Ferrum:** `KvCache` in `layer.rs` and `LlamaCache` in `llm.rs`.

### Decoding Strategy
The rule for picking the next token from the model's probability distribution.

### Greedy Decoding
Always pick the single most likely token. Deterministic and simple, but often
repetitive and dull.

### Sampling
Pick the next token at random, weighted by the probabilities. Produces more
varied, natural text.

### Temperature
A setting that reshapes the probabilities before sampling. Low temperature
(e.g. 0.2) makes the model focused and predictable; high temperature (e.g. 1.2)
makes it more creative and more error-prone. Temperature 0 is effectively
greedy decoding.

### Top-k Sampling
Only sample from the *k* most likely tokens (for example the top 40), ignoring
the long tail of unlikely ones.

### Top-p Sampling (Nucleus Sampling)
Only sample from the smallest set of tokens whose probabilities add up to *p*
(for example 0.9). The set grows when the model is unsure and shrinks when it
is confident.

### Min-p Sampling
Only keep tokens whose probability is at least some fraction of the top
token's probability. A newer alternative to top-p.

### Repetition Penalty
Lowers the score of tokens that have already appeared, to stop the model from
looping on the same phrase. Especially useful for small models.
**In Ferrum:** these settings live in `SamplingParams` in `slm.rs`.

### Beam Search
Keep several candidate continuations at once and pick the one with the best
overall probability. Common in translation; rarely used for chat.

### Max Tokens and Stop Sequences
**Max tokens** caps how long a reply can be. A **stop sequence** is text that,
once generated, ends the reply (for example the EOS token or `"\nUser:"`).

### Latency, Throughput, TTFT, Tokens per Second
- **Latency**: how long one request takes.
- **Time to first token (TTFT)**: the wait before the first word appears;
  mostly the prefill phase.
- **Tokens per second**: how fast the reply streams during decode.
- **Throughput**: total tokens per second across all users at once.

### Batching (Inference)
Processing several users' requests together to use hardware more efficiently.
**Continuous batching** adds and removes requests on the fly.

### Streaming
Sending tokens to the user as they are generated instead of waiting for the
whole reply.

### Speculative Decoding
A speed-up where a small, fast **draft model** guesses several tokens ahead and
the big model checks them all in one pass, keeping the ones it agrees with. The
output is the same as the big model alone, just faster.

### Hallucination
When a model states something false or invented with confidence, such as a fake
citation. It happens because the model generates *plausible* text, not
verified facts. Smaller models hallucinate more on general knowledge.

---

## 8. Making models small and fast

### Quantization
Storing weights (and sometimes activations) with fewer bits, for example 8-bit
or 4-bit integers instead of 32-bit floats. A 4× to 8× smaller model needs less
memory, loads faster, and runs faster on CPUs, usually with only a small loss
in quality.
**In Ferrum:** `ferrum_core/src/quant.rs`.

### int8 and int4
Quantization to 8-bit integers (256 possible values, ¼ the size of FP32) or
4-bit integers (16 possible values, ⅛ the size). int4 is the most common
setting for running LLMs on laptops.

### Scale and Zero-Point
How quantization maps real numbers onto integers. The **scale** says how much
one integer step is worth; the **zero-point** says which integer stands for
0.0. To get a weight back: `real ≈ scale × (integer − zero_point)`.

### Block-wise (Group-wise) Quantization
Splitting weights into small blocks (for example 32 weights) and giving each
block its own scale. One outlier weight then only harms its own block, which
greatly improves accuracy.

### Post-Training Quantization (PTQ)
Quantizing a model *after* it is fully trained. Fast and needs no training, but
can lose more quality, especially at very low bit widths.

### Quantization-Aware Training (QAT)
Training the model while simulating quantization, so it learns weights that
still work well after they are rounded to integers. Better quality than PTQ at
the same size.
**In Ferrum:** int8 QAT is built into the trainer.

### Fake Quantization
The trick behind QAT: during the forward pass, round weights to their quantized
values and immediately convert them back to floats. The model "feels" the
rounding error while training stays in floating point.

### GGUF Quantization Types (Q8_0, Q4_0, Q4_K_M, …)
The naming scheme used by llama.cpp. The number is roughly the bits per weight.
- **Q8_0, Q4_0, Q4_1**: simple block quantization (blocks of 32 weights).
- **K-quants (Q2_K to Q6_K)**: smarter schemes using larger "super-blocks"
  with nested scales, for better quality at the same size.
- Suffixes **_S, _M, _L** (small, medium, large) mix precisions, keeping
  sensitive layers at higher precision. **Q4_K_M** is a popular balance of size
  and quality.

### Pruning
Removing weights, neurons, attention heads, or whole layers that contribute
little, making the model smaller and faster.

### Memory Footprint
How much RAM a model needs. The weights take roughly
`parameters × bytes per parameter`, plus room for the KV cache and working
memory. See the table in [Rules of thumb](#12-rules-of-thumb).

### Memory-Bandwidth Bound vs Compute Bound
A task is **compute bound** when the processor's arithmetic speed is the limit,
and **memory-bandwidth bound** when the limit is how fast data can be moved
from RAM. Token-by-token decoding is usually memory-bandwidth bound, which is
why smaller (quantized) weights make it faster.

### Kernel
A small, highly optimised routine for one operation, such as matmul or softmax.
Inference speed depends heavily on how good the kernels are.

### BLAS
"Basic Linear Algebra Subprograms": a standard library interface for fast
vector and matrix maths (OpenBLAS, Intel MKL, Apple Accelerate). Most ML
frameworks call a BLAS for their matmuls.
**In Ferrum:** deliberately not used; Ferrum writes its own kernels for
transparency.

### SIMD
"Single Instruction, Multiple Data": CPU instructions that do the same
operation on several numbers at once (AVX2 and AVX-512 on Intel/AMD, NEON on
ARM). A major source of CPU inference speed.

### Multi-threading and Thread Pool
Splitting work across several CPU cores. A **thread pool** keeps worker threads
alive and ready, avoiding the cost of creating new ones for every operation.
**In Ferrum:** `ferrum_core/src/parallel.rs`; set the count with
`FERRUM_NUM_THREADS`.

### CPU, GPU, NPU
A **CPU** is a general-purpose processor with a few powerful cores. A **GPU**
has thousands of simpler cores and very fast memory, ideal for the parallel
maths of large models. An **NPU** (neural processing unit) is a chip built
specifically for AI, now common in phones and laptops.

### Edge AI / On-device AI
Running models directly on the user's device (phone, laptop, sensor) rather
than in the cloud. Benefits: privacy, offline use, no per-request cost, and low
latency. This is the main home of SLMs.

### WebAssembly (WASM)
A portable binary format that runs at near-native speed inside web browsers.
Compiling a model runtime to WASM lets an SLM run in a browser tab with no
server.
**In Ferrum:** the `slm_wasm` crate.

---

## 9. Files, formats, and tools

### Model File (Weights File)
The file that stores a trained model's parameters, and often its configuration
and tokenizer.

### Safetensors
A safe, fast file format for model weights, standard on Hugging Face. Unlike
older Python "pickle" files, it cannot hide executable code.

### GGUF
A single-file model format from the llama.cpp project. It packs the weights
(often quantized), the architecture settings, and the tokenizer into one file.
The standard format for running models locally on CPUs.
**In Ferrum:** read in `gguf.rs`, written in `gguf_write.rs`.

### GGML
The C tensor library behind llama.cpp, and the name of GGUF's predecessor
format.

### FINF
Ferrum's own self-contained model format: weights, metadata, and tokenizer in
one `.bin` file.
**In Ferrum:** `ferrum_core/src/loader.rs`; details in `Manual.md`.

### ONNX
"Open Neural Network Exchange": a format for moving models between frameworks
and running them with optimised runtimes such as ONNX Runtime.

### llama.cpp
A popular open-source C/C++ program for running LLMs efficiently on CPUs and
consumer GPUs, using GGUF files.

### Ollama and LM Studio
Desktop apps that make running local models easy. **Ollama** is
command-line-first with a local API server; **LM Studio** has a graphical chat
interface. Both use llama.cpp and GGUF underneath.

### Hugging Face (Hub, Transformers)
The main website for sharing models and datasets (the **Hub**), and the
company's widely used Python library (**Transformers**) for loading and
training them.

### PyTorch, TensorFlow, JAX
The main Python deep-learning frameworks. PyTorch is the most used for
language model research and training.

### CUDA
NVIDIA's platform for programming its GPUs. Most GPU training depends on it.

### Model Card
A document published with a model describing what it is for, how it was
trained, how it was evaluated, its limits, and its licence.

### Model Families (Llama, Qwen, Mistral, Gemma, Phi)
Well-known open-weight model series from Meta (Llama), Alibaba (Qwen), Mistral
AI (Mistral), Google (Gemma), and Microsoft (Phi). Most come in several sizes,
including small ones suitable as SLMs.

### Model Licence
The legal terms for using a model. Some are permissive (Apache 2.0, MIT); some
restrict commercial use or very large deployments. Always check before
shipping a product.

---

## 10. Building applications with LLMs

### Prompt Engineering
Designing the wording and structure of prompts to get better, more reliable
outputs.

### Zero-shot and Few-shot Prompting
**Zero-shot**: ask the model to do a task with no examples. **Few-shot**:
include a few worked examples in the prompt so the model copies the pattern.

### In-Context Learning
A model's ability to pick up a task from instructions and examples in the
prompt alone, without any change to its weights.

### Chain-of-Thought (CoT)
Getting a model to write out its reasoning step by step before giving an
answer. This improves accuracy on maths and logic problems.

### Reasoning Model
A model trained to "think" at length (often in hidden reasoning tokens) before
answering. More accurate on hard problems, but slower and more expensive per
answer.

### Embedding Model
A model that turns a whole piece of text into a single vector, so that texts
with similar meanings get similar vectors. Used for search and RAG.

### Vector Database
A database built to store embedding vectors and quickly find the ones most
similar to a query vector. Examples: FAISS, Chroma, pgvector, Pinecone.

### Semantic Search
Search by meaning rather than exact words. A search for "car repair" can find
a document about "fixing automobiles".

### RAG (Retrieval-Augmented Generation)
Fetching relevant documents (usually by semantic search) and putting them in
the prompt so the model answers from that material. It reduces hallucination
and lets a model use private or up-to-date knowledge without retraining. A
small model plus good retrieval can beat a big model without it.

### Chunking
Splitting long documents into smaller pieces before embedding them for RAG, so
each piece fits in the context window and stays on one topic.

### Reranking
A second, more careful pass that reorders retrieved chunks by relevance before
they go into the prompt.

### Function Calling / Tool Use
Letting a model ask a program to run a tool (a calculator, a web search, a
database query) by producing a structured request. The program runs the tool
and returns the result to the model.

### Agent
A system where a model works in a loop: it plans, calls tools, looks at the
results, and decides what to do next until a goal is reached.

### Structured Output (JSON Mode)
Forcing a model's output to follow a fixed format such as valid JSON, so other
programs can read it reliably.

### Guardrails
Checks around a model that block or fix unsafe, off-topic, or badly formatted
inputs and outputs.

### Prompt Injection and Jailbreak
**Prompt injection**: hidden instructions inside content the model reads (a web
page, an email) that try to hijack its behaviour. **Jailbreak**: a prompt
crafted to make a model ignore its safety rules. Both are key security risks
for LLM applications.

---

## 11. Evaluating models

### Evaluation (Evals)
Measuring how well a model performs, using test sets, benchmarks, or human
judgement. Build evals for *your* task early; general benchmarks rarely tell
you how a model will do on your specific job.

### Benchmark
A standard test set used to compare models. Common ones:
- **MMLU**: multiple-choice questions across 57 school and professional
  subjects.
- **HellaSwag**: choosing the most sensible ending to a short scenario.
- **GSM8K**: grade-school maths word problems.
- **HumanEval**: writing Python functions that pass unit tests.

### Held-out Data
Data the model never saw during training, used to measure how well it
*generalises* to new examples.

### Data Contamination
When benchmark questions leak into the training data, so the model has seen the
answers and its scores are inflated.

### Accuracy, BLEU, ROUGE
**Accuracy**: the share of answers that are exactly right. **BLEU** and
**ROUGE**: scores that measure word overlap between generated text and a
reference text; used for translation and summarisation, but they miss meaning.

### LLM-as-a-Judge
Using a strong model to grade another model's outputs. Fast and cheap compared
with human review, but the judge has its own biases.

### Human Evaluation
People rating or comparing model outputs. Slow and costly, but still the gold
standard for open-ended quality.

### Ablation
Removing or changing one part of a system and measuring the effect, to find out
which parts actually matter.

---

## 12. Rules of thumb

**Memory needed just to hold the weights:**

| Precision | Bytes per parameter | 100M model | 1B model | 7B model |
|---|---|---|---|---|
| FP32 | 4 | 400 MB | 4 GB | 28 GB |
| FP16 / BF16 | 2 | 200 MB | 2 GB | 14 GB |
| int8 (Q8) | ~1 | ~100 MB | ~1 GB | ~7 GB |
| int4 (Q4) | ~0.5 | ~50 MB | ~0.5 GB | ~3.5 GB |

Add extra room for the KV cache and working memory. Training needs far more:
with AdamW in FP32, plan for roughly **16 bytes per parameter** (weights,
gradients, and two optimizer states).

**Other handy numbers:**
- 1 token ≈ ¾ of an English word; 1,000 tokens ≈ 750 words.
- Training compute ≈ `6 × parameters × tokens` FLOPs.
- Compute-optimal training data ≈ 20 tokens per parameter (Chinchilla); SLMs
  are often trained on far more.
- On a CPU, decode speed is roughly `memory bandwidth ÷ model size in bytes`
  tokens per second, which is why 4-bit models generate much faster than FP32.
