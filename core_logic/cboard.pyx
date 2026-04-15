# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.pair cimport pair

np.import_array()

# ─── C++ declarations ───
cdef extern from "bitboard.h":
    cdef cppclass Bitset225:
        Bitset225() except +
        void set(int idx)
        void clear(int idx)
        bool test(int idx)

    cdef cppclass Board:
        Board() except +
        void set_stone(int x, int y, int player)
        void remove_stone(int x, int y)
        void undo()
        void reset()
        bool is_empty(int x, int y)
        bool is_black(int x, int y)
        bool is_white(int x, int y)
        int stone_at(int x, int y)
        bool check_win(int player)
        bool is_full()
        int current_player()
        void set_current_player(int p)
        int num_stones()
        vector[pair[int,int]] get_legal_moves()
        bool is_forbidden(int x, int y)
        void get_state(float out[3][225])
        void get_symmetries(const float state[3][225], const float policy[225],
                            float out_states[8][3][225], float out_policies[8][225])


cdef class PyBoard:
    """Python wrapper around C++ Board for 15x15 Gomoku with Renju rules."""
    cdef Board* _board

    def __cinit__(self):
        self._board = new Board()

    def __dealloc__(self):
        del self._board

    def copy(self):
        """Deep copy of the board."""
        cdef PyBoard b = PyBoard()
        b._board.set_stone(0, 0, 0)  # just to init; we'll use get_state
        b._board = new Board()
        # Copy state via stone iteration
        cdef Board* src = self._board
        cdef int x, y, s
        for y in range(15):
            for x in range(15):
                s = src.stone_at(x, y)
                if s == 1:
                    b._board.set_stone(x, y, 1)
                elif s == 2:
                    b._board.set_stone(x, y, 2)
        b._board.set_current_player(src.current_player())
        return b

    def reset(self):
        self._board.reset()

    def play_move(self, int x, int y, int player):
        """Place a stone. player: 1=black, 2=white."""
        self._board.set_stone(x, y, player)

    def undo_move(self):
        self._board.undo()

    def is_empty(self, int x, int y):
        return self._board.is_empty(x, y)

    def is_black(self, int x, int y):
        return self._board.is_black(x, y)

    def is_white(self, int x, int y):
        return self._board.is_white(x, y)

    def stone_at(self, int x, int y):
        return self._board.stone_at(x, y)

    def check_win(self, int player):
        return self._board.check_win(player)

    def is_full(self):
        return self._board.is_full()

    @property
    def current_player(self):
        return self._board.current_player()

    @current_player.setter
    def current_player(self, int p):
        self._board.set_current_player(p)

    @property
    def num_stones(self):
        return self._board.num_stones()

    def legal_moves(self):
        """Return list of (x, y) legal moves for the current player."""
        cdef vector[pair[int,int]] moves = self._board.get_legal_moves()
        return [(moves[i].first, moves[i].second) for i in range(moves.size())]

    def is_forbidden(self, int x, int y):
        """Check if placing black at (x,y) is forbidden."""
        return self._board.is_forbidden(x, y)

    def get_state(self):
        """Return state as numpy array of shape (3, 15, 15) float32.
        Planes: [current_player_stones, opponent_stones, color_to_move]."""
        cdef np.ndarray[np.float32_t, ndim=3] out = np.zeros((3, 15, 15), dtype=np.float32)
        cdef float state[3][225]
        self._board.get_state(state)
        for c in range(3):
            for i in range(225):
                out[c, i // 15, i % 15] = state[c][i]
        return out

    def get_legal_moves_mask(self):
        """Return boolean mask of shape (15, 15) for legal moves."""
        cdef np.ndarray[np.uint8_t, ndim=2] mask = np.zeros((15, 15), dtype=np.uint8)
        cdef vector[pair[int,int]] moves = self._board.get_legal_moves()
        cdef int i
        for i in range(moves.size()):
            mask[moves[i].second, moves[i].first] = 1  # row=y, col=x
        return mask

    def symmetries(self, np.ndarray[np.float32_t, ndim=3] state not None,
                   np.ndarray[np.float32_t, ndim=1] policy not None):
        """Generate 8 symmetric (state, policy) pairs.
        Returns list of (state, policy) tuples."""
        cdef float c_state[3][225]
        cdef float c_policy[225]
        cdef float out_states[8][3][225]
        cdef float out_policies[8][225]

        # Copy numpy arrays into C arrays
        for c in range(3):
            for i in range(225):
                c_state[c][i] = state[c, i // 15, i % 15]
        for i in range(225):
            c_policy[i] = policy[i]

        self._board.get_symmetries(c_state, c_policy, out_states, out_policies)

        result = []
        cdef np.ndarray[np.float32_t, ndim=3] s
        cdef np.ndarray[np.float32_t, ndim=1] p
        for t in range(8):
            s = np.zeros((3, 15, 15), dtype=np.float32)
            p = np.zeros(225, dtype=np.float32)
            for c in range(3):
                for i in range(225):
                    s[c, i // 15, i % 15] = out_states[t][c][i]
            for i in range(225):
                p[i] = out_policies[t][i]
            result.append((s, p))
        return result
