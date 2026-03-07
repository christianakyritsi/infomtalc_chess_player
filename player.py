import random
import chess
import torch
from typing import Optional
from minicons import scorer

from chess_tournament.players import Player

class TransformerPlayer(Player):
    # Basic material values used for evaluating board positions
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    def __init__(self, name: str = "TransformerPlayer"):
        # Initialize base player class
        super().__init__(name)

        # Language model used to score chess move sequences
        self.model_name = "gpt2"

        # Use GPU if available, otherwise fall back to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # IncrementalLMScorer allows scoring text continuations with gpt2
        # Here I use it to evaluate how "natural" a move looks after a fen prompt
        self.scorer = scorer.IncrementalLMScorer(self.model_name, device=self.device)

        # Beam sizes for pruning search
        # These control how many moves it keeps at each depth
        self.K0 = 14 # candidate moves at root
        self.K1 = 10 # opp responses
        self.K2 = 8 # our replies

        # Weight of language model score at root move
        self.lm_weight_root = 0.02

        # kept small so lm does not dominate chess evaluation
        self.lm_weight_reply = 0.005

        # Penalty if we repeat the previous move 
        self.undo_penalty = 0.35

        # Penalty if our move allows the opponent to capture a major piece
        self.hang_major_penalty = 2.5

        # Track last move played
        self.last_move = None

    # Checks whether our move leaves a rook or queen hanging
    # If the opponent can capture one of these pieces immediately, apply a penalty
    def major_hang_penalty(self, board_after_my_move, my_color):
        pen = 0.0
        for opp in board_after_my_move.legal_moves:
            if board_after_my_move.is_capture(opp):
                captured = board_after_my_move.piece_at(opp.to_square)
                if captured and captured.color == my_color and captured.piece_type in (chess.QUEEN, chess.ROOK):
                    pen -= self.hang_major_penalty
        return pen

    # Simple evaluation function based only on material balance. Positive values mean advantage for the player
    def eval_material(self, board, my_color):

        # Losing positions are extremely bad
        if board.is_checkmate():
            return -1e6

        # Draw positions get neutral score
        if board.is_stalemate():
            return 0

        val = 0

        # Count material difference
        for pt, v in self.PIECE_VALUES.items():
            val += v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))

        # Convert to perspective of current player
        return val if my_color == chess.WHITE else -val

    # Heuristic move ordering. Ranks moves using simple rules: captures are good, checks are good. Then keep the top-k moves
    def prune(self, board, moves, k):
        scored = []
        for m in moves:
            score = 0

            # Captures are strongly prioritized
            if board.is_capture(m):
                captured = board.piece_at(m.to_square)
                if captured:
                    score += 10 * self.PIECE_VALUES[captured.piece_type]

            # Give bonus if the move gives check
            board.push(m)
            if board.is_check():
                score += 2
            board.pop()
            scored.append((score, m))

        # Sort moves by heuristic score
        scored.sort(reverse=True, key=lambda x: x[0])

        # Return only the top-k moves
        return [m for _, m in scored[: min(k, len(scored))]]

    #  Main decision function. Performs a shallow minimax search with pruning and adds language model scoring at the root
    def choose_move(self, fen):
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        my_color = board.turn

        # Mate in 1
        for m in legal:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                self.last_move = m.uci()
                return m.uci()
            board.pop()

        # Prune root moves
        root = self.prune(board, legal, self.K0)

        # Create prompt for language model
        prompt = f"FEN: {fen}\nMove:"

        # Score candidate moves using gpt2
        lm_scores = self.scorer.sequence_score([prompt + " " + m.uci() for m in root])

        best_move = root[0]
        best_val = -1e18

        # Evaluate each candidate move
        for i, my in enumerate(root):
            b1 = chess.Board(fen)
            b1.push(my)

            # Check if the move hangs a major piece
            hang_pen = self.major_hang_penalty(b1, my_color)

            # Generate opponent responses
            opp_moves = list(b1.legal_moves)
            opp_pruned = self.prune(b1, opp_moves, self.K1)

            worst_case = 1e18

            # Simulate opponent replies
            for opp in opp_pruned:
                b2 = chess.Board(fen)
                b2.push(my)
                b2.push(opp)

                # Generate our replies
                replies = self.prune(b2, list(b2.legal_moves), self.K2)

                best_reply = -1e18
                for r in replies:
                    b3 = chess.Board(b2.fen())
                    b3.push(r)
                    val = self.eval_material(b3, my_color)
                    best_reply = max(best_reply, val)

                # Opponent will choose move minimizing our score
                worst_case = min(worst_case, best_reply)

            # Combine search score + lm score + penalties
            total = worst_case + self.lm_weight_root * lm_scores[i] + hang_pen

            if total > best_val:
                best_val = total
                best_move = my

        self.last_move = best_move.uci()
        return best_move.uci()

    def get_move(self, fen: str) -> Optional[str]:
        try:
            return self.choose_move(fen)
        except Exception:
            return None
