import torch
import chess

#TODO: Try to make this thing fully vectorized

def board2vec(board: chess.Board, device: torch.device) -> torch.Tensor:
    tensor = torch.zeros((8, 8, 12), device=device)

    piece_map = board.piece_map()

    squares = torch.tensor(list(piece_map.keys()), device=device)
    pieces = torch.tensor([piece.piece_type for piece in piece_map.values()], device=device)
    colors = torch.tensor([piece.color for piece in piece_map.values()], device=device)

    ranks = squares // 8
    files = squares % 8

    # Map pieces to channels: 0-5 (white), 6-11 (black)
    # P N B R Q K - p n b r q k
    channels = pieces - 1 + colors * 6  

    # one-hot encode, 0 for empty squares
    tensor[ranks, files, channels] = 1

    return tensor
