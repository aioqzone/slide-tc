from typing import Tuple

from .jigsaw import Jigsaw, imitate_drag

__all__ = ["solve_slide_captcha", "Jigsaw", "imitate_drag"]


def solve_slide_captcha(
    background: bytes, piece: bytes, init_pos: Tuple[int, int]
) -> int:
    return Jigsaw(background, piece, init_pos).solve()
