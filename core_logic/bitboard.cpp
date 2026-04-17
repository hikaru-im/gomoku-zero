#include "bitboard.h"
#include <cstring>
#include <algorithm>

// ─── Bitset225 implementation ───

Bitset225 Bitset225::shr(int n) const {
    Bitset225 r;
    if (n <= 0) { r = *this; r.mask225(); return r; }
    if (n >= 256) return r;
    for (int i = 0; i < BB_WORDS; i++) {
        int lo = n;       // bits shifting out of word i (from top)
        int hi = 64 - n;  // bits staying in word i
        if (n < 64) {
            r.w[i] = (w[i] >> lo);
            if (i + 1 < BB_WORDS)
                r.w[i] |= (w[i + 1] << hi);
        } else {
            // entire word is shifted out
            int src_word = i + (n >> 6);
            int bit_off  = n & 63;
            if (src_word < BB_WORDS) {
                r.w[i] = w[src_word] >> bit_off;
                if (bit_off > 0 && src_word + 1 < BB_WORDS)
                    r.w[i] |= (w[src_word + 1] << (64 - bit_off));
            }
        }
    }
    r.mask225();
    return r;
}

void Bitset225::mask225() {
    // word 3 (bits 192-255): only bits 192-224 are valid (33 bits)
    w[3] &= (1ULL << (225 - 192)) - 1ULL; // keep bits 0..32 of word 3
}

// ─── Board: static mask initialization ───

bool Board::s_masks_ready = false;
Bitset225 Board::s_col[BOARD_SIZE];
Bitset225 Board::s_can_shr[4][5]; // [dir_index][k]

void Board::init_masks() {
    if (s_masks_ready) return;

    // Column masks
    for (int c = 0; c < BOARD_SIZE; c++)
        for (int r = 0; r < BOARD_SIZE; r++)
            s_col[c].set(r * BOARD_SIZE + c);

    // For each direction and shift count k (1..4),
    // compute mask of positions that can safely be shifted right
    for (int d = 0; d < 4; d++) {
        for (int k = 0; k < 5; k++) {
            if (k == 0) {
                // no shift needed — all bits valid
                for (int i = 0; i < BOARD_CELLS; i++) s_can_shr[d][0].set(i);
                continue;
            }
            s_can_shr[d][k] = Bitset225(); // zero
            for (int y = 0; y < BOARD_SIZE; y++) {
                for (int x = 0; x < BOARD_SIZE; x++) {
                    int idx = y * BOARD_SIZE + x;
                    // After shifting right by step*k, this bit must land on a valid
                    // neighbor in the same direction. That means the source (what
                    // we're reading from) must be at offset (+k*step) from here
                    // and be a valid board cell.
                    // Equivalent: position (x + k*dx, y + k*dy) must be on board,
                    // where (dx,dy) is the direction vector.
                    int sx, sy;
                    if (d == 0) { sx = x + k; sy = y; }           // H: step right
                    else if (d == 1) { sx = x; sy = y + k; }      // V: step down
                    else if (d == 2) { sx = x + k; sy = y + k; }  // D1: SE
                    else { sx = x + k; sy = y - k; }              // D2: NE (step=14 means row decreases)

                    if (sx >= 0 && sx < BOARD_SIZE && sy >= 0 && sy < BOARD_SIZE)
                        s_can_shr[d][k].set(idx);
                }
            }
        }
    }

    s_masks_ready = true;
}

// ─── Board constructor ───

Board::Board() : current_player_(1) {
    init_masks();
    black_ = Bitset225();
    white_ = Bitset225();
}

void Board::reset() {
    black_ = Bitset225();
    white_ = Bitset225();
    current_player_ = 1;
    move_history_.clear();
}

void Board::copy_from(const Board& other) {
    black_ = other.black_;
    white_ = other.white_;
    current_player_ = other.current_player_;
    move_history_ = other.move_history_;
}

// ─── Stone management ───

void Board::set_stone(int x, int y, int player) {
    int idx = y * BOARD_SIZE + x;
    if (player == 1) black_.set(idx);
    else             white_.set(idx);
    move_history_.push_back({x, y, player});
    current_player_ = (player == 1) ? 2 : 1;
}

void Board::remove_stone(int x, int y) {
    int idx = y * BOARD_SIZE + x;
    black_.clear(idx);
    white_.clear(idx);
}

void Board::undo() {
    if (move_history_.empty()) return;
    Move& m = move_history_.back();
    remove_stone(m.x, m.y);
    current_player_ = m.player;
    move_history_.pop_back();
}

int Board::stone_at(int x, int y) const {
    if (black_.test(y * 15 + x)) return 1;
    if (white_.test(y * 15 + x)) return 2;
    return 0;
}

bool Board::is_full() const {
    Bitset225 occupied = black_ | white_;
    return occupied.count() >= BOARD_CELLS;
}

// ─── Win detection (bitwise) ───

bool Board::has_five(const Bitset225& stones) const {
    int steps[] = {STEP_H, STEP_V, STEP_D1, STEP_D2};
    for (int d = 0; d < 4; d++) {
        Bitset225 a = stones;
        for (int k = 1; k <= 4; k++) {
            Bitset225 shifted = stones & s_can_shr[d][k];
            a = a & shifted.shr(steps[d] * k);
        }
        if (a.any()) return true;
    }
    return false;
}

bool Board::has_exactly_five(const Bitset225& stones) const {
    // Find positions where there are 5+ in a row, then verify no 6th stone
    int steps[] = {STEP_H, STEP_V, STEP_D1, STEP_D2};
    for (int d = 0; d < 4; d++) {
        Bitset225 a = stones;
        for (int k = 1; k <= 4; k++) {
            Bitset225 shifted = stones & s_can_shr[d][k];
            a = a & shifted.shr(steps[d] * k);
        }
        // For each candidate 5-in-a-row position, check for 6th stone
        if (a.any()) {
            for (int idx = 0; idx < BOARD_CELLS; idx++) {
                if (!a.test(idx)) continue;
                // Found a 5-in-a-row starting at idx in direction d
                // Check: is there a 6th stone before or after?
                int step = steps[d];
                int before = idx - step;
                int after = idx + 5 * step;
                bool has_before = (before >= 0 && stones.test(before));
                bool has_after = (after < BOARD_CELLS && stones.test(after));
                if (!has_before && !has_after) return true;
            }
        }
    }
    return false;
}

bool Board::has_five_plus(const Bitset225& stones) const {
    // For white: 5 or more in a row is a win
    // Check 5-in-a-row (which already implies >= 5 since we AND 5 aligned positions)
    return has_five(stones);
}

bool Board::check_win(int player) const {
    if (player == 1) {
        // Black wins with exactly 5 (overline = forbidden, not a win)
        return has_exactly_five(black_);
    } else {
        // White wins with 5 or more
        return has_five(white_);
    }
}

// ─── Legal moves ───

std::vector<std::pair<int,int>> Board::get_legal_moves() {
    std::vector<std::pair<int,int>> moves;
    Bitset225 occupied = black_ | white_;
    bool check_forbidden = (current_player_ == 1 && black_.count() >= 3);
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            int idx = y * BOARD_SIZE + x;
            if (!occupied.test(idx)) {
                if (check_forbidden && is_forbidden(x, y))
                    continue;
                moves.push_back({x, y});
            }
        }
    }
    return moves;
}

// ─── Line extraction ───

void Board::extract_line(int x, int y, int dx, int dy,
                         int cells[BOARD_SIZE], int& len) const {
    // Walk backward to find the start of the relevant segment
    int sx = x, sy = y;
    while (sx - dx >= 0 && sx - dx < BOARD_SIZE &&
           sy - dy >= 0 && sy - dy < BOARD_SIZE) {
        sx -= dx; sy -= dy;
    }

    len = 0;
    int cx = sx, cy = sy;
    while (cx >= 0 && cx < BOARD_SIZE && cy >= 0 && cy < BOARD_SIZE) {
        cells[len++] = stone_at(cx, cy);
        cx += dx; cy += dy;
    }
}

// ─── Overline detection ───

bool Board::has_overline_at(int x, int y) {
    // Temporarily place black at (x,y) and check for 6+ in a row
    int idx = y * BOARD_SIZE + x;
    black_.set(idx);
    bool result = false;

    // Check all 4 directions: count consecutive black stones through (x,y)
    int dirs[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    for (auto& d : dirs) {
        int dx = d[0], dy = d[1];
        int count = 1;
        // Count in positive direction
        for (int i = 1; i <= 5; i++) {
            int nx = x + i*dx, ny = y + i*dy;
            if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && is_black(nx, ny))
                count++;
            else break;
        }
        // Count in negative direction
        for (int i = 1; i <= 5; i++) {
            int nx = x - i*dx, ny = y - i*dy;
            if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && is_black(nx, ny))
                count++;
            else break;
        }
        if (count >= 6) { result = true; break; }
    }

    black_.clear(idx);
    return result;
}

// ─── Count open threes at (x,y) ───

int Board::count_open_threes(int x, int y) {
    // Place black temporarily
    int idx = y * BOARD_SIZE + x;
    black_.set(idx);
    int open_three_count = 0;

    int dirs[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    for (auto& d : dirs) {
        int dx = d[0], dy = d[1];

        // Extract line of length up to 11 centered on (x,y)
        // We need 5 cells on each side to check open-three patterns
        int line[11];
        int cx = x - 5*dx, cy = y - 5*dy;
        for (int i = 0; i < 11; i++) {
            if (cx >= 0 && cx < BOARD_SIZE && cy >= 0 && cy < BOARD_SIZE) {
                line[i] = stone_at(cx, cy);
            } else {
                line[i] = 2; // treat off-board as white (blocking)
            }
            cx += dx; cy += dy;
        }

        // The placed stone is at position 5 in the line.
        // Check for open-three patterns within this window.
        // An "open three" is a configuration where black has exactly 3 stones
        // in a 5-cell span, with both ends empty, and adding one stone creates
        // an open four (which is 4 in 5 with both ends empty).

        // Scan all 5-cell windows that contain position 5 (the placed stone)
        bool found_open_three = false;
        for (int start = 0; start <= 6 && !found_open_three; start++) {
            if (start > 5 || start + 5 < 5) continue; // window must contain pos 5

            int blacks = 0, whites = 0, empties = 0;
            int empty_positions[5];
            int ep_count = 0;

            for (int i = 0; i < 5; i++) {
                int pos = start + i;
                if (line[pos] == 1) blacks++;
                else if (line[pos] == 2) whites++;
                else { empties++; empty_positions[ep_count++] = i; }
            }

            // Need exactly 3 black, 2 empty in the window
            if (blacks != 3 || empties != 2) continue;

            // At minimum, the 5-window has both endpoints empty
            if (line[start] != 0 || line[start + 4] != 0) continue;

            // Now check: can one of the two empty positions become an open four?
            // An open four is 4 black in a 5-window with both ends empty.
            for (int e = 0; e < ep_count && !found_open_three; e++) {
                int test_line[5];
                memcpy(test_line, &line[start], 5);
                test_line[empty_positions[e]] = 1; // fill one empty with black

                int tb = 0, te = 0;
                for (int i = 0; i < 5; i++) {
                    if (test_line[i] == 1) tb++;
                    else if (test_line[i] == 0) te++;
                }

                if (tb == 4 && te == 1) {
                    // 4 black + 1 empty in a 5-window
                    // Check if it's an open four: both sides outside are empty
                    int tl = (start > 0) ? line[start - 1] : 2;
                    int tr = (start + 5 < 11) ? line[start + 5] : 2;
                    if (tl == 0 && tr == 0) {
                        // This is an open three!
                        found_open_three = true;
                    }
                }
            }
        }

        if (found_open_three) open_three_count++;
    }

    black_.clear(idx);
    return open_three_count;
}

// ─── Count fours at (x,y) ───

int Board::count_fours(int x, int y) {
    int idx = y * BOARD_SIZE + x;
    black_.set(idx);
    int four_count = 0;

    int dirs[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    for (auto& d : dirs) {
        int dx = d[0], dy = d[1];

        // Extract line of length 11 centered on (x,y)
        int line[11];
        int cx = x - 5*dx, cy = y - 5*dy;
        for (int i = 0; i < 11; i++) {
            if (cx >= 0 && cx < BOARD_SIZE && cy >= 0 && cy < BOARD_SIZE)
                line[i] = stone_at(cx, cy);
            else
                line[i] = 2; // off-board = blocking
            cx += dx; cy += dy;
        }

        bool found_four = false;
        // Scan all 5-cell windows and 6-cell windows containing position 5
        // A "four" is any pattern where one more stone completes five-in-a-row.
        // This includes:
        //   Solid: B B B B _  or  _ B B B B
        //   Split: B B _ B B  or  B _ B B B  or  B B B _ B

        // Check all 5-cell windows containing position 5
        for (int start = 0; start <= 6 && !found_four; start++) {
            if (start > 5 || start + 5 < 5) continue;

            int blacks = 0, whites = 0, empties = 0;
            for (int i = 0; i < 5; i++) {
                int pos = start + i;
                if (line[pos] == 1) blacks++;
                else if (line[pos] == 2) whites++;
                else empties++;
            }

            if (blacks == 4 && empties == 1 && whites == 0) {
                found_four = true;
            }
        }

        // Also check 6-cell windows for split fours like B B _ B B
        if (!found_four) {
            for (int start = 0; start <= 5 && !found_four; start++) {
                if (start > 5 || start + 6 < 5) continue;

                int blacks = 0, whites = 0, empties = 0;
                for (int i = 0; i < 6; i++) {
                    int pos = start + i;
                    if (line[pos] == 1) blacks++;
                    else if (line[pos] == 2) whites++;
                    else empties++;
                }

                if (blacks == 4 && empties == 2 && whites == 0) {
                    // Check if one stone fills both empties? No, that's impossible.
                    // Check if filling one empty creates 5 in a row within this window
                    for (int i = 0; i < 6 && !found_four; i++) {
                        int pos = start + i;
                        if (line[pos] != 0) continue;

                        // Fill this empty with black and check for 5-in-a-row
                        int temp_line[6];
                        memcpy(temp_line, &line[start], 6);
                        temp_line[i] = 1;

                        // Check all 5-cell sub-windows of this 6-cell window
                        for (int s = 0; s <= 1; s++) {
                            int b5 = 0, w5 = 0;
                            for (int j = 0; j < 5; j++) {
                                if (temp_line[s + j] == 1) b5++;
                                else if (temp_line[s + j] == 2) w5++;
                            }
                            if (b5 == 5) { found_four = true; break; }
                        }
                    }
                }
            }
        }

        if (found_four) four_count++;
    }

    black_.clear(idx);
    return four_count;
}

// ─── Combined forbidden check ───

bool Board::is_forbidden(int x, int y) {
    // Only black has forbidden moves
    if (current_player_ != 1) return false;
    // Must be empty
    if (!is_empty(x, y)) return false;

    // If placing here creates five-in-a-row for black, it's NEVER forbidden
    // (winning move overrides all forbidden rules)
    int idx = y * BOARD_SIZE + x;
    black_.set(idx);
    bool makes_five = has_five(black_);
    black_.clear(idx);
    if (makes_five) return false;

    // Check overline (6+)
    if (has_overline_at(x, y)) return true;

    // Check 4-4
    if (count_fours(x, y) >= 2) return true;

    // Check 3-3
    if (count_open_threes(x, y) >= 2) return true;

    return false;
}

// ─── State export ───

void Board::get_state(float out[3][225]) const {
    const Bitset225* cur  = (current_player_ == 1) ? &black_ : &white_;
    const Bitset225* opp  = (current_player_ == 1) ? &white_ : &black_;

    for (int i = 0; i < 225; i++) {
        out[0][i] = cur->test(i) ? 1.0f : 0.0f;
        out[1][i] = opp->test(i) ? 1.0f : 0.0f;
        out[2][i] = (current_player_ == 1) ? 1.0f : 0.0f; // color-to-move plane
    }
}

// ─── Symmetry augmentation (D4 group) ───

// Transform indices for each of 8 symmetries:
// 0: identity
// 1: rotate 90 CW
// 2: rotate 180
// 3: rotate 270 CW
// 4: flip horizontal
// 5: flip horizontal + rotate 90 CW
// 6: flip horizontal + rotate 180
// 7: flip horizontal + rotate 270 CW

static int transform_coord(int x, int y, int transform) {
    int N = BOARD_SIZE - 1; // 14
    switch (transform) {
        case 0: return y * BOARD_SIZE + x;             // identity
        case 1: return x * BOARD_SIZE + (N - y);       // rot 90 CW
        case 2: return (N - y) * BOARD_SIZE + (N - x); // rot 180
        case 3: return (N - x) * BOARD_SIZE + y;       // rot 270 CW
        case 4: return y * BOARD_SIZE + (N - x);       // flip H
        case 5: return (N - x) * BOARD_SIZE + (N - y); // flip H + rot 90
        case 6: return (N - y) * BOARD_SIZE + x;       // flip H + rot 180
        case 7: return x * BOARD_SIZE + y;             // flip H + rot 270
        default: return y * BOARD_SIZE + x;
    }
}

void Board::get_symmetries(const float state[3][225], const float policy[225],
                            float out_states[8][3][225], float out_policies[8][225]) {
    for (int t = 0; t < 8; t++) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            for (int x = 0; x < BOARD_SIZE; x++) {
                int src_idx = y * BOARD_SIZE + x;
                int dst_idx = transform_coord(x, y, t);

                for (int c = 0; c < 3; c++)
                    out_states[t][c][dst_idx] = state[c][src_idx];
                out_policies[t][dst_idx] = policy[src_idx];
            }
        }
    }
}
