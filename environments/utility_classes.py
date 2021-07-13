from typing import NamedTuple


class MovementProperties(NamedTuple):
    allow_square_movement: bool = True
    allow_diagonal_movement: bool = False
    allow_no_op: bool = False
