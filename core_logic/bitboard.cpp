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
                    else { sx = x - k; sy = y + k; }              // D2: step=14 goes SW (dx=-1,dy=+1)

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
    // Direction vectors for each step (the direction of idx + step)
    int dx[] = {1, 0, 1, -1};   // H:right, V:down, D1:SE, D2:SW
    int dy[] = {0, 1, 1, 1};

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
                int sx = idx % BOARD_SIZE;
                int sy = idx / BOARD_SIZE;
                // Check 6th stone before start and after end using coordinates
                int bx = sx - dx[d], by = sy - dy[d];
                bool has_before = (bx >= 0 && bx < BOARD_SIZE &&
                                   by >= 0 && by < BOARD_SIZE &&
                                   stones.test(by * BOARD_SIZE + bx));
                int ax = sx + 5 * dx[d], ay = sy + 5 * dy[d];
                bool has_after = (ax >= 0 && ax < BOARD_SIZE &&
                                  ay >= 0 && ay < BOARD_SIZE &&
                                  stones.test(ay * BOARD_SIZE + ax));
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

// ─── Helper: check if placing black at (px,py) creates exactly five ───
// Assumes (x,y) is already placed (caller manages that).
// (px,py) must differ from (x,y).

bool Board::would_be_exactly_five(int px, int py) {
    int pidx = py * BOARD_SIZE + px;
    black_.set(pidx);
    bool result = has_exactly_five(black_);
    black_.clear(pidx);
    return result;
}

// ─── Helper: check if placing black at (px,py) creates exactly five
//  in a specific direction. Used for straight four detection.

bool Board::would_be_exactly_five_in_dir(int px, int py, int dx, int dy) {
    int pidx = py * BOARD_SIZE + px;
    black_.set(pidx);
    // Count consecutive through (px,py) in direction (dx,dy)
    int count = 1;
    for (int i = 1; i <= 4; i++) {
        int nx = px + i*dx, ny = py + i*dy;
        if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && is_black(nx, ny))
            count++;
        else break;
    }
    for (int i = 1; i <= 4; i++) {
        int nx = px - i*dx, ny = py - i*dy;
        if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && is_black(nx, ny))
            count++;
        else break;
    }
    black_.clear(pidx);
    if (count != 5) return false;
    // Verify no 6th stone on either end
    int ax = px + 5*dx, ay = py + 5*dy;
    bool has_after = (ax >= 0 && ax < BOARD_SIZE && ay >= 0 && ay < BOARD_SIZE && is_black(ax, ay));
    int bx = px - 5*dx, by = py - 5*dy;
    bool has_before = (bx >= 0 && bx < BOARD_SIZE && by >= 0 && by < BOARD_SIZE && is_black(bx, by));
    return !has_after && !has_before;
}

// ─── Count fours at (x,y) — strict Renju definition ───
// A "four" = a configuration where one more black stone creates exactly five.
// After placing black at (x,y), for each direction, check if any empty point
// on the line can complete exactly five.

int Board::count_fours(int x, int y) {
    int idx = y * BOARD_SIZE + x;
    black_.set(idx);
    int four_count = 0;

    int dirs[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    for (auto& d : dirs) {
        int dx = d[0], dy = d[1];
        bool found_four = false;

        // Scan empty points within ±5 along the line
        for (int dist = -5; dist <= 5 && !found_four; dist++) {
            if (dist == 0) continue;
            int px = x + dist * dx, py = y + dist * dy;
            if (px < 0 || px >= BOARD_SIZE || py < 0 || py >= BOARD_SIZE) continue;
            if (!is_empty(px, py)) continue;

            // Try placing black at (px,py) and check for exactly five
            if (would_be_exactly_five(px, py)) {
                found_four = true;
            }
        }

        if (found_four) four_count++;
    }

    black_.clear(idx);
    return four_count;
}

// ─── Helper: shallow forbidden check (overline + double-four only, no double-three) ───
// Used by live-three detection to break recursion at depth 1.

bool Board::is_forbidden_no_three(int px, int py) {
    if (!is_empty(px, py)) return false;

    int pidx = py * BOARD_SIZE + px;
    black_.set(pidx);

    // Exactly five → not forbidden
    if (has_exactly_five(black_)) {
        black_.clear(pidx);
        return false;
    }

    // Overline → forbidden
    black_.clear(pidx);
    if (has_overline_at(px, py)) return true;

    // Double-four → forbidden
    if (count_fours(px, py) >= 2) return true;

    // Do NOT check double-three (this breaks recursion)
    return false;
}

// ─── Count live threes at (x,y) — strict Renju definition ───
// A "live three" = a configuration where placing one more black stone (at a legal
// non-forbidden point, without immediately making five) creates a straight four.
// A "straight four" = 4 consecutive blacks with empty on both sides, where either
// side can complete exactly five.

int Board::count_live_threes(int x, int y) {
    int idx = y * BOARD_SIZE + x;
    black_.set(idx);
    int three_count = 0;

    int dirs[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    for (auto& d : dirs) {
        int dx = d[0], dy = d[1];
        bool found_live_three = false;

        // Scan empty points within ±5 along the line
        for (int dist = -5; dist <= 5 && !found_live_three; dist++) {
            if (dist == 0) continue;
            int px = x + dist * dx, py = y + dist * dy;
            if (px < 0 || px >= BOARD_SIZE || py < 0 || py >= BOARD_SIZE) continue;
            if (!is_empty(px, py)) continue;

            // Placing at (px,py) must NOT immediately make five
            // (if it makes five, this is already a four, not a three)
            black_.set(py * BOARD_SIZE + px);
            bool makes_five = has_exactly_five(black_);
            black_.clear(py * BOARD_SIZE + px);
            if (makes_five) continue;

            // Check if (px,py) is forbidden (shallow: overline + double-four only)
            // If forbidden, this extension is illegal → not a live three
            if (is_forbidden_no_three(px, py)) continue;

            // Check if placing at (px,py) creates a straight four in this direction
            // A straight four = 4 consecutive blacks with both neighbors empty,
            // and filling either neighbor creates exactly five.
            black_.set(py * BOARD_SIZE + px);

            // Find 4-consecutive-black segment through (px,py) in this direction
            int count = 1;
            int ex = px + dx, ey = py + dy;
            while (ex >= 0 && ex < BOARD_SIZE && ey >= 0 && ey < BOARD_SIZE && is_black(ex, ey)) {
                count++;
                ex += dx; ey += dy;
            }
            // (ex, ey) is now the cell AFTER the consecutive run in positive direction
            int bx = px - dx, by = py - dy;
            while (bx >= 0 && bx < BOARD_SIZE && by >= 0 && by < BOARD_SIZE && is_black(bx, by)) {
                count++;
                bx -= dx; by -= dy;
            }
            // (bx, by) is now the cell BEFORE the consecutive run in negative direction

            if (count == 4) {
                // Check both ends are empty and can create exactly five
                bool pos_end_ok = (ex >= 0 && ex < BOARD_SIZE && ey >= 0 && ey < BOARD_SIZE)
                                  && is_empty(ex, ey)
                                  && would_be_exactly_five_in_dir(ex, ey, dx, dy);
                bool neg_end_ok = (bx >= 0 && bx < BOARD_SIZE && by >= 0 && by < BOARD_SIZE)
                                  && is_empty(bx, by)
                                  && would_be_exactly_five_in_dir(bx, by, dx, dy);
                if (pos_end_ok && neg_end_ok) {
                    found_live_three = true;
                }
            }

            black_.clear(py * BOARD_SIZE + px);
        }

        if (found_live_three) three_count++;
    }

    black_.clear(idx);
    return three_count;
}

// ─── Combined forbidden check ───

bool Board::is_forbidden(int x, int y) {
    // Only black has forbidden moves
    if (current_player_ != 1) return false;
    // Must be empty
    if (!is_empty(x, y)) return false;

    // If placing here creates exactly five-in-a-row for black, it's NEVER forbidden
    // (winning move overrides all forbidden rules; overline is still forbidden)
    int idx = y * BOARD_SIZE + x;
    black_.set(idx);
    bool makes_five = has_exactly_five(black_);
    black_.clear(idx);
    if (makes_five) return false;

    // Check overline (6+)
    if (has_overline_at(x, y)) return true;

    // Check 4-4
    if (count_fours(x, y) >= 2) return true;

    // Check 3-3 (with recursive legality verification)
    if (count_live_threes(x, y) >= 2) return true;

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
