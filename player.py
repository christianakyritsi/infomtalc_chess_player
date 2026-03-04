import random
import chess
import torch
from typing import Optional
from minicons import scorer

from chess_tournament.players import Player

class TransformerPlayer(Player):
    
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)
        
        self.model_name = "gpt2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scorer = scorer.IncrementalLMScorer(self.model_name, device=self.device)

        self.K0 = 14
        self.K1 = 10
        self.K2 = 8

        self.lm_weight_root = 0.02
        self.lm_weight_reply = 0.005

        self.undo_penalty = 0.35
        self.hang_major_penalty = 2.5

        self.last_move = None

    def major_hang_penalty(self, board_after_my_move, my_color):
        pen = 0.0
        for opp in board_after_my_move.legal_moves:
            if board_after_my_move.is_capture(opp):
                captured = board_after_my_move.piece_at(opp.to_square)
                if captured and captured.color == my_color and captured.piece_type in (chess.QUEEN, chess.ROOK):
                    pen -= self.hang_major_penalty
        return pen

    def eval_material(self, board, my_color):
        if board.is_checkmate():
            return -1e6
        if board.is_stalemate():
            return 0

        val = 0
        for pt, v in self.PIECE_VALUES.items():
            val += v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))

        return val if my_color == chess.WHITE else -val

    def prune(self, board, moves, k):
        scored = []
        for m in moves:
            score = 0
            if board.is_capture(m):
                captured = board.piece_at(m.to_square)
                if captured:
                    score += 10 * self.PIECE_VALUES[captured.piece_type]
            board.push(m)
            if board.is_check():
                score += 2
            board.pop()
            scored.append((score, m))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored[: min(k, len(scored))]]

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

        root = self.prune(board, legal, self.K0)

        prompt = f"FEN: {fen}\nMove:"
        lm_scores = self.scorer.sequence_score([prompt + " " + m.uci() for m in root])

        best_move = root[0]
        best_val = -1e18

        for i, my in enumerate(root):
            b1 = chess.Board(fen)
            b1.push(my)

            hang_pen = self.major_hang_penalty(b1, my_color)

            opp_moves = list(b1.legal_moves)
            opp_pruned = self.prune(b1, opp_moves, self.K1)

            worst_case = 1e18

            for opp in opp_pruned:
                b2 = chess.Board(fen)
                b2.push(my)
                b2.push(opp)

                replies = self.prune(b2, list(b2.legal_moves), self.K2)

                best_reply = -1e18
                for r in replies:
                    b3 = chess.Board(b2.fen())
                    b3.push(r)
                    val = self.eval_material(b3, my_color)
                    best_reply = max(best_reply, val)

                worst_case = min(worst_case, best_reply)

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
