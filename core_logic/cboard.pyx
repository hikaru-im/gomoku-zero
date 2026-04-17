# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair

np.import_array()

cdef extern from "bitboard.h":
    cdef cppclass Board:
        Board() except +
        void set_stone(int x, int y, int player)
        void remove_stone(int x, int y)
        void undo()
        void reset()
        void copy_from(const Board& other)
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


cdef class PyBoard:
    cdef Board* _board

    def __cinit__(self):
        self._board = new Board()

    def __dealloc__(self):
        if self._board is not NULL:
            del self._board

    def copy(self):
        cdef PyBoard b = PyBoard()
        b._board.copy_from(self._board[0])
        return b

    def reset(self):
        self._board.reset()

    def play_move(self, int x, int y, int player):
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
        cdef vector[pair[int,int]] moves = self._board.get_legal_moves()
        cdef list result = []
        cdef Py_ssize_t i, n = moves.size()
        for i in range(n):
            result.append((moves[i].first, moves[i].second))
        return result

    def is_forbidden(self, int x, int y):
        return self._board.is_forbidden(x, y)

    def get_state(self):
        cdef np.ndarray[np.float32_t, ndim=3] out = np.zeros((3, 15, 15), dtype=np.float32)
        cdef float state[3][225]
        cdef int c, i
        self._board.get_state(state)
        for c in range(3):
            for i in range(225):
                out[c, i // 15, i % 15] = state[c][i]
        return out

    def get_legal_moves_mask(self):
        cdef np.ndarray[np.uint8_t, ndim=2] mask = np.zeros((15, 15), dtype=np.uint8)
        cdef vector[pair[int,int]] moves = self._board.get_legal_moves()
        cdef Py_ssize_t i, n = moves.size()
        for i in range(n):
            mask[moves[i].second, moves[i].first] = 1
        return mask

    def symmetries(self, np.ndarray[np.float32_t, ndim=3] state not None,
                   np.ndarray[np.float32_t, ndim=1] policy not None):
        cdef list result = []
        cdef np.ndarray[np.float32_t, ndim=2] board_policy = policy.reshape(15, 15)
        cdef object s
        cdef object p
        cdef int k

        for k in range(4):
            s = np.rot90(state, k=k, axes=(1, 2)).copy()
            p = np.rot90(board_policy, k=k).copy().reshape(225)
            result.append((s.astype(np.float32), p.astype(np.float32)))

        state_flipped = np.flip(state, axis=2)
        policy_flipped = np.flip(board_policy, axis=1)
        for k in range(4):
            s = np.rot90(state_flipped, k=k, axes=(1, 2)).copy()
            p = np.rot90(policy_flipped, k=k).copy().reshape(225)
            result.append((s.astype(np.float32), p.astype(np.float32)))

        return result
