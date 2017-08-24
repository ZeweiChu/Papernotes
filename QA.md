
# Question Answering

## Zewei Chu

## Summaries

## In ACL 2017

## [Question Answering through Transfer Learning from Large Fine-grained Supervision Data](http://aclweb.org/anthology/P/P17/P17-2081.pdf)


### Dataset
- Pretrain on SQuAD, evaluate on WikiQA and SemEval 2016 (Task 3A)
- WikiQA: classify whether each sentence provides the answer to the query
- SemEval: classify whether each comment is relevant to the question.
- SICK: recognizing textual entailment (RTE)


### Model
[BiDAF](https://allenai.github.io/bi-att-flow/ https://arxiv.org/pdf/1611.01603.pdf)

### Experiment
![Results](QATransfer.png)

### Comments

---

### [Coarse-to-Fine Question Answering for Long Documents](http://aclweb.org/anthology/P/P17/P17-1020.pdf)

- fast model for selecting relevant sentences
- expensive RNN for producing the answer from those sentences.
- sentence selection as a latent variable trained jointly from the answer only using reinforcement learning.
- Wiki reading dataset https://www.aclweb.org/anthology/P/P16/P16-1145.pdf


---

## [An End-to-End Model for Question Answering over Knowledge Base with Cross-Attention Combining Global Knowledge](http://aclweb.org/anthology/P/P17/P17-1021.pdf)

represent the questions and their corresponding scores dynamically according to the various candidate answer aspects via cross-attention mechanism.
Contributions: 1. a novel cross-attention based NN model tailored to KB-QA task, which considers the mutual influence between the representation of questions and the corresponding answer aspects. 2. We leverage the global KB information, aiming at represent the answers more precisely. It also alleviates the OOV problem, which is very helpful to the cross-attention model. 3. The experimental results on the open dataset WebQuestions demonstrate the effectiveness of the proposed approach. 

### Model
- First, we identify the topic entity of the question, and generate candidate answers from Freebase. 
- Then, a cross-attention based neural network is employed to represent the question under the influence of the candidate answer aspects. 
- Finally, the similarity score between the question and each corresponding candidate answer is calculated, and the candidates with highest score will be selected as the final answers

### Experiment
Freebase API to select candidate answers. Introduce answer-towards-question attention and question-towards-answer attention. 


## [Learning to Ask: Neural Question Generation for Reading Comprehension](http://aclweb.org/anthology/P/P17/P17-1123.pdf)

attention-based sequence learning model for the task and investigate the effect of encoding sentence- vs. paragraph-level information




---

## [Search-based Neural Structured Learning for Sequential Question Answering](http://aclweb.org/anthology/P/P17/P17-1167.pdf)

Task: answering sequences of simple but inter-related questions. semantic parsing for answering sequences of simple related questions
Dataset: WikiTableQuestions(http://www.cs.stanford.edu/people/ppasupat/resource/ACL2015-paper.pdf : answering complex questions on semi-structured tables using question-answer pairs as supervision.)




---

## [Reading Wikipedia to Answer Open-Domain Questions](http://aclweb.org/anthology/P/P17/P17-1171.pdf)

---
## [Improved Neural Relation Detection for Knowledge Base Question Answering](http://aclweb.org/anthology/P/P17/P17-1053.pdf)

---
## [End-to-End Non-Factoid Question Answering with an Interactive Visualization of Neural Attention Weights](http://aclweb.org/anthology/P/P17/P17-4004.pdf)



