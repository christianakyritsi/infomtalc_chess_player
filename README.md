# ♟️ GPT-2 Hybrid Chess Bot

A transformer-based chess agent that combines GPT-2 language modeling with rule-based search and chess heuristics to generate competitive moves.

---

## 🚀 Overview

This project explores whether a language model (GPT-2) can be used to guide decision-making in a structured domain like chess.

Instead of relying purely on traditional chess engines, the model evaluates moves based on how “natural” they appear given a board state, while ensuring legality and competitiveness through search and heuristics.

---

## 🧠 Approach

The system combines three key components:

### 1. Transformer-Based Scoring

* Uses GPT-2 to score candidate moves as text sequences
* Board state is encoded as a FEN string prompt
* Moves are ranked based on language model likelihood

### 2. Search + Pruning

* Performs a shallow minimax-style search
* Uses heuristic pruning to reduce the search space:

  * Prioritizes captures and checks
  * Keeps top-k moves at each depth

### 3. Rule-Based Heuristics

* Material evaluation (piece values)
* Penalty for repeating moves
* Penalty for “hanging” major pieces (rook/queen)

---

## ⚙️ Key Features

* Hybrid model combining LLMs and classical search
* Custom pruning strategy (multi-level beam search)
* Lightweight evaluation without full engine dependency
* Robust fallback handling for invalid outputs

---

## 📊 Evaluation

The agent was evaluated against multiple baselines:

* Random player
* Greedy player
* Language-model-only agent
* Engine-based opponents

🏆 **Result: Ranked 6th out of 81 participants**

The model consistently outperformed baseline agents, including simplified engine-based approaches.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* GPT-2 (via `minicons`)
* python-chess

---

## 📂 Project Structure

* `player.py` — main implementation of the hybrid agent

---

## 💡 Key Insight

This project demonstrates that language models can be used beyond text generation, acting as probabilistic priors in structured decision-making tasks when combined with domain constraints.

---

## ▶️ Run the Project

You can run the notebook in Google Colab:

[(https://colab.research.google.com/drive/1C2T2r-MX_n-mEDLb1Ptd298Fx0WzSCcF)]

---

## 📌 Future Improvements

* Deeper search (alpha-beta pruning)
* Fine-tuning GPT-2 on chess move data
* Integration with stronger evaluation functions
* Hybridization with reinforcement learning

---

## 👤 Author

Christiana Kyritsi
MSc Applied Data Science — Utrecht University
