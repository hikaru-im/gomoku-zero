#ifndef BITBOARD_H
#define BITBOARD_H

#include <cstdint>
#include <vector>
#include <string>

static constexpr int BOARD_SIZE = 15;
static constexpr int BOARD_CELLS = 225;
static constexpr int BB_WORDS = 4; // 4 x uint64_t = 256 bits > 225

// ─── Compact bitboard for 15x15 board ───
struct Bitset225 {
    uint64_t w[BB_WORDS];

    Bitset225() : w{0, 0, 0, 0} {}

    inline void set(int idx) {
        w[idx >> 6] |= (1ULL << (idx & 63));
    }

    inline void clear(int idx) {
        w[idx >> 6] &= ~(1ULL << (idx & 63));
    }

    inline bool test(int idx) const {
        return (w[idx >> 6] >> (idx & 63)) & 1ULL;
    }

    inline Bitset225 operator&(const Bitset225& o) const {
        Bitset225 r;
        for (int i = 0; i < BB_WORDS; i++) r.w[i] = w[i] & o.w[i];
        return r;
    }

    inline Bitset225 operator|(const Bitset225& o) const {
        Bitset225 r;
        for (int i = 0; i < BB_WORDS; i++) r.w[i] = w[i] | o.w[i];
        return r;
    }

    inline Bitset225 operator~() const {
        Bitset225 r;
        for (int i = 0; i < BB_WORDS; i++) r.w[i] = ~w[i];
        return r;
    }

    inline Bitset225 operator&=(const Bitset225& o) {
        for (int i = 0; i < BB_WORDS; i++) w[i] &= o.w[i];
        return *this;
    }

    inline bool any() const { return w[0] | w[1] | w[2] | w[3]; }
    inline bool none() const { return !any(); }

    inline int count() const {
        int c = 0;
        for (int i = 0; i < BB_WORDS; i++) c += __builtin_popcountll(w[i]);
        return c;
    }

    // Right-shift by n bits with mask225 cleanup
    Bitset225 shr(int n) const;
    void mask225(); // clear bits >= 225
};

// Direction step sizes for 1D index
static constexpr int STEP_H  = 1;
static constexpr int STEP_V  = 15;
static constexpr int STEP_D1 = 16; // diagonal backslash  (SE)
static constexpr int STEP_D2 = 14; // diagonal slash      (NE)

// ─── Main Board class ───
class Board {
public:
    Board();

    // --- Stone management ---
    void set_stone(int x, int y, int player);   // player: 1=black, 2=white
    void remove_stone(int x, int y);
    void undo();
    void reset();

    // --- Queries ---
    inline bool is_empty(int x, int y) const {
        return !black_.test(y * 15 + x) && !white_.test(y * 15 + x);
    }
    inline bool is_black(int x, int y) const { return black_.test(y * 15 + x); }
    inline bool is_white(int x, int y) const { return white_.test(y * 15 + x); }
    int  stone_at(int x, int y) const; // 0=empty, 1=black, 2=white

    // --- Game state ---
    bool check_win(int player) const; // player: 1=black, 2=white
    bool is_full() const;
    int  current_player() const { return current_player_; }
    void set_current_player(int p) { current_player_ = p; }
    int  num_stones() const { return (int)move_history_.size(); }

    // --- Legal moves (forbidden-filtered for black) ---
    std::vector<std::pair<int,int>> get_legal_moves() const;

    // --- Forbidden-move detection (black only) ---
    bool is_forbidden(int x, int y) const;

    // --- State export for neural net ---
    // Output: [3][225] float32 — current player stones, opponent stones, color-to-move plane
    void get_state(float out[3][225]) const;

    // --- Symmetry augmentation (D4 group, 8 transforms) ---
    static void get_symmetries(const float state[3][225], const float policy[225],
                               float out_states[8][3][225], float out_policies[8][225]);

private:
    Bitset225 black_, white_;
    int current_player_; // 1 = black to move, 2 = white to move

    struct Move { int x, y, player; };
    std::vector<Move> move_history_;

    // --- Precomputed masks (initialized once) ---
    static bool   s_masks_ready;
    static Bitset225 s_col[BOARD_SIZE];     // column bitmasks
    static Bitset225 s_can_shr[4][5];       // s_can_shr[dir][k]: safe to shift right by step[dir]*k

    static void init_masks();

    // --- Win detection (bitwise) ---
    bool has_five(const Bitset225& stones) const;  // exactly 5-in-a-row (for black: 5 only)
    bool has_five_plus(const Bitset225& stones) const; // 5+ in a row (for white)

    // --- Forbidden helpers ---
    bool has_overline_at(int x, int y) const;   // 6+ in a row at (x,y) after placing black
    int  count_open_threes(int x, int y) const;  // number of open-3 lines created
    int  count_fours(int x, int y) const;        // number of four lines created

    // Line extraction: fill cells[] with stone states along direction through (x,y)
    // 0=empty, 1=black, 2=white.  len is number of valid cells (up to 15)
    void extract_line(int x, int y, int dx, int dy,
                      int cells[BOARD_SIZE], int& len) const;
};

#endif // BITBOARD_H
