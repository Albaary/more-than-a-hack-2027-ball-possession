#!/usr/bin/env python3
"""
predict_possession.py — Two-pass, rule-based ball possession pipeline.
More Than A Hack 2027 (Phase 1: Ball Possession). Team: Alamak Menang Wak.

Streams the tactical tracking file (< 15 MB RAM) and labels every frame H / A / D / X:
  1. Ball trajectory, ball speed, missing-ball gaps (interpolated only up to 35 frames).
  2. Nearest-player distances per team and mean player speed.
  3. Smart Tagging context: team prior and Set Piece dead/alive windows.
  4. Separate team (H vs A) and dead-ball (D) tracks, smoothed independently.
  5. Segment clean-up, boundary snapping and latency correction.

See README.md for parameters and TECHNICAL.md for the possession rules.

Usage, from the repository root:
    python predict_possession.py --team "Alamak Menang Wak"
"""

import argparse
import os
import shutil
import sys
import numpy as np

# Pastikan root repositori selalu ada di sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import (
    HOME,
    AWAY,
    NOBODY,
    FIELD_SEP,
    GROUP_SEP,
    MISSING,
    team_of_each_track,
    team_names,
    write_submission,
)
from starter.load_smart_tagging import load as load_smart_tagging

# Rentang sebelum kick-off babak 2 (110808..113741) ikut dinilai dan wajib 'D' (FAQ.md)
SECOND_HALF_PRE_KICKOFF = (110808, 113741)


def _moving_average_1d(arr, half_win):
    """Rata-rata bergerak 1D cepat menggunakan cumulative sum (mengabaikan NaN)."""
    valid = ~np.isnan(arr)
    vals = np.where(valid, arr, 0.0).astype(np.float64)
    c_val = np.concatenate(([0.0], np.cumsum(vals)))
    c_cnt = np.concatenate(([0], np.cumsum(valid.astype(np.int32))))
    n = len(arr)
    idx = np.arange(n)
    low = np.maximum(0, idx - half_win)
    high = np.minimum(n, idx + half_win + 1)
    cnt = c_cnt[high] - c_cnt[low]
    out = np.full(n, np.nan, dtype=np.float32)
    mask = cnt > 0
    out[mask] = ((c_val[high[mask]] - c_val[low[mask]]) / cnt[mask]).astype(np.float32)
    return out


def stream_atd_two_pass(raw_path, team_of_track, n_frames, pitch):
    """
    Membaca BAR-ATH_tactical_atd.txt dalam 2 pass streaming hemat memori (< 15 MB RAM):
    - Pass 1: Posisi bola mentah, kecepatan bola (m/s), durasi gap, kedekatan garis tepi,
              dan interpolasi bola khusus untuk occlusion pendek (<= 35 frame / 1.4 detik).
    - Pass 2: Jarak pemain terdekat ke-1 & ke-2 per tim terhadap bola + kecepatan rata-rata pemain.
    """
    length, width = pitch
    ball_x_raw = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y_raw = np.full(n_frames, np.nan, dtype=np.float32)

    print(f"[1/5] Pass 1/2: Streaming lintasan bola dari {os.path.basename(raw_path)}...")
    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            sep1 = line.find(FIELD_SEP)
            sep2 = line.rfind(FIELD_SEP)
            if sep1 < 0 or sep2 <= sep1:
                continue

            ball = line[sep2 + 1 :].strip()
            if not ball or ball[0] == MISSING:
                continue

            frame_idx = int(line[:sep1]) - 1
            if frame_idx < 0 or frame_idx >= n_frames:
                continue

            c1 = ball.find(",")
            c2 = ball.find(",", c1 + 1)
            if c1 < 0 or c2 < 0:
                continue
            ball_x_raw[frame_idx] = float(ball[:c1]) * length
            ball_y_raw[frame_idx] = float(ball[c1 + 1 : c2]) * width

    # Hitung kecepatan bola instan (m/s) pada frame terlihat
    ball_speed_raw = np.full(n_frames, np.nan, dtype=np.float32)
    dx_b = np.diff(ball_x_raw)
    dy_b = np.diff(ball_y_raw)
    step_speed = np.hypot(dx_b, dy_b) * 25.0
    valid_step = ~np.isnan(step_speed) & (step_speed < 45.0)
    ball_speed_raw[1:][valid_step] = step_speed[valid_step]
    ball_speed = _moving_average_1d(ball_speed_raw, half_win=3)

    # Hitung panjang beruntun setiap gap bola hilang (ball_missing_raw)
    ball_missing_raw = np.isnan(ball_x_raw) | np.isnan(ball_y_raw)
    gap_len = np.zeros(n_frames, dtype=np.int32)
    gap_near_boundary = np.zeros(n_frames, dtype=bool)

    near_margin = 2.8  # Meter dari garis pinggir / gawang
    is_near_edge = (
        ~ball_missing_raw
        & (
            (ball_x_raw < near_margin)
            | (ball_x_raw > length - near_margin)
            | (ball_y_raw < near_margin)
            | (ball_y_raw > width - near_margin)
        )
    )

    i = 0
    while i < n_frames:
        if ball_missing_raw[i]:
            j = i
            while j < n_frames and ball_missing_raw[j]:
                j += 1
            g_len = j - i
            gap_len[i:j] = g_len
            left_edge = is_near_edge[i - 1] if i > 0 else False
            right_edge = is_near_edge[j] if j < n_frames else False
            if left_edge or right_edge:
                gap_near_boundary[i:j] = True
            i = j
        else:
            i += 1

    # Interpolasi posisi bola HANYA untuk occlusion pendek (<= 35 frame / 1.4 detik)
    # agar lintasan lurus imajiner pada gap panjang tidak memotong bek lawan
    ball_x = ball_x_raw.copy()
    ball_y = ball_y_raw.copy()
    short_gap = ball_missing_raw & (gap_len <= 35)
    all_idx = np.arange(n_frames)
    valid_b = ~ball_missing_raw
    if np.any(valid_b):
        ball_x[short_gap] = np.interp(all_idx[short_gap], all_idx[valid_b], ball_x_raw[valid_b]).astype(
            np.float32
        )
        ball_y[short_gap] = np.interp(all_idx[short_gap], all_idx[valid_b], ball_y_raw[valid_b]).astype(
            np.float32
        )

    # Pass 2: Jarak pemain ke-1 & ke-2 terdekat ke bola serta rata-rata kecepatan pemain
    dist_h1 = np.full(n_frames, np.nan, dtype=np.float32)
    dist_h2 = np.full(n_frames, np.nan, dtype=np.float32)
    dist_a1 = np.full(n_frames, np.nan, dtype=np.float32)
    dist_a2 = np.full(n_frames, np.nan, dtype=np.float32)
    raw_player_speed = np.full(n_frames, np.nan, dtype=np.float32)

    n_slots = len(team_of_track)
    graded_mask = np.zeros(n_frames, dtype=bool)
    for _, s_f, e_f in half_video_blocks(GAME):
        graded_mask[s_f : e_f + 1] = True

    print("[1/5] Pass 2/2: Menghitung jarak pemain ke bola & kecepatan kolektif...")
    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            sep1 = line.find(FIELD_SEP)
            if sep1 < 0:
                continue
            frame_idx = int(line[:sep1]) - 1
            if frame_idx < 0 or frame_idx >= n_frames or not graded_mask[frame_idx]:
                continue

            sep2 = line.rfind(FIELD_SEP, sep1 + 1)
            if sep2 < 0:
                continue

            bx = ball_x[frame_idx]
            by = ball_y[frame_idx]
            has_ball = not (np.isnan(bx) or np.isnan(by))

            h1 = 999999.0
            h2 = 999999.0
            a1 = 999999.0
            a2 = 999999.0
            speed_sum = 0.0
            speed_cnt = 0

            players_str = line[sep1 + 1 : sep2]
            for slot, group in enumerate(players_str.split(GROUP_SEP)):
                if slot >= n_slots or not group or group[0] == MISSING:
                    continue
                gc1 = group.find(",")
                gc2 = group.find(",", gc1 + 1)
                if gc1 < 0 or gc2 < 0:
                    continue

                try:
                    ps = float(group[gc2 + 1 :])
                    if 0.0 <= ps < 15.0:
                        speed_sum += ps
                        speed_cnt += 1
                except ValueError:
                    pass

                if has_ball:
                    dx = float(group[:gc1]) * length - bx
                    dy = float(group[gc1 + 1 : gc2]) * width - by
                    d2 = dx * dx + dy * dy
                    if team_of_track[slot] == HOME:
                        if d2 < h1:
                            h2 = h1
                            h1 = d2
                        elif d2 < h2:
                            h2 = d2
                    else:
                        if d2 < a1:
                            a2 = a1
                            a1 = d2
                        elif d2 < a2:
                            a2 = d2

            if speed_cnt > 0:
                raw_player_speed[frame_idx] = speed_sum / speed_cnt
            if has_ball:
                if h1 < 900000.0:
                    dist_h1[frame_idx] = float(np.sqrt(h1))
                if h2 < 900000.0:
                    dist_h2[frame_idx] = float(np.sqrt(h2))
                if a1 < 900000.0:
                    dist_a1[frame_idx] = float(np.sqrt(a1))
                if a2 < 900000.0:
                    dist_a2[frame_idx] = float(np.sqrt(a2))

    # Interpolasi juga dist_h1 & dist_a1 pada gap pendek <= 35 frame jika masih ada NaN
    for arr in (dist_h1, dist_h2, dist_a1, dist_a2):
        valid_d = ~np.isnan(arr)
        if np.any(valid_d):
            arr[short_gap] = np.interp(all_idx[short_gap], all_idx[valid_d], arr[valid_d]).astype(
                np.float32
            )

    mean_player_speed = _moving_average_1d(raw_player_speed, half_win=12)
    seen = int(np.sum(~ball_missing_raw))
    print(f"      Bola terlihat pada {seen}/{n_frames} frame ({100.0 * seen / n_frames:.1f}%).")
    return (
        ball_x_raw,
        ball_y_raw,
        ball_x,
        ball_y,
        ball_speed,
        ball_missing_raw,
        gap_len,
        gap_near_boundary,
        dist_h1,
        dist_h2,
        dist_a1,
        dist_a2,
        mean_player_speed,
    )


def extract_smart_tagging_context(
    n_frames,
    pitch,
    ball_x_raw,
    ball_y_raw,
    ball_speed,
    ball_missing_raw,
    gap_len,
    mean_player_speed,
    dist_h1,
    dist_a1,
    game=GAME,
    data_dir=None,
):
    """
    Memetakan Smart Tagging menjadi:
    - prior_team: Prior fase ofensif (+6 frame latency alignment) + transisi bertahan singkat.
    - dead_ball_window: Jendela bola mati sebelum Set Piece dieksekusi.
    - alive_restart_window: Jendela bola hidup tepat setelah Set Piece dieksekusi.
    """
    print("[2/5] Memetakan konteks Smart Tagging & Set Pieces...")
    prior_team = np.full(n_frames, NOBODY, dtype=np.int8)
    dead_ball_window = np.zeros(n_frames, dtype=bool)
    alive_restart_window = np.zeros(n_frames, dtype=bool)

    try:
        events = load_smart_tagging(game=game, data_dir=data_dir)
    except Exception as exc:
        print(f"      [Info] Smart Tagging dilewati ({exc}).")
        return prior_team, dead_ball_window, alive_restart_window

    length, width = pitch
    offensive_codes = {
        "BUILD UP",
        "PROGRESSION",
        "CREATING CHANCES",
        "ATTACKING TRANSITION",
        "BALL IN FINAL THIRD",
        "BALL IN THE BOX",
        "LONG BALL",
        "GOALS",
    }

    set_pieces = []

    # Langkah A: Pasang DEFENSIVE TRANSITION singkat (maks 60 frame awal) hanya pada area kosong,
    # lalu timpa dengan seluruh fase ofensif (+6 frame latency alignment)
    for pass_type in ("def_trans", "offensive"):
        for ev in events:
            code = (ev.get("code") or "").strip().upper()
            s_frame = max(0, min(n_frames - 1, int(ev.get("start_frame", 0))))
            e_frame = max(0, min(n_frames - 1, int(ev.get("end_frame", 0))))
            tags = ev.get("tags") or {}

            team_str = (tags.get("Team") or tags.get("team") or "").strip().lower()
            t_code = (
                HOME
                if "barcelona" in team_str
                else (AWAY if "atletic" in team_str or "athletic" in team_str else NOBODY)
            )

            if pass_type == "def_trans" and code == "DEFENSIVE TRANSITION" and t_code != NOBODY:
                opp_code = AWAY if t_code == HOME else HOME
                s_eff = min(e_frame, s_frame + 6)
                e_eff = min(e_frame, s_eff + 60)
                seg_slice = prior_team[s_eff : e_eff + 1]
                prior_team[s_eff : e_eff + 1] = np.where(seg_slice == NOBODY, opp_code, seg_slice)
            elif pass_type == "offensive":
                if code == "SET PIECES":
                    set_pieces.append((s_frame, e_frame, t_code, tags))
                elif code in offensive_codes and t_code != NOBODY:
                    s_eff = min(e_frame, s_frame + 6)
                    prior_team[s_eff : e_frame + 1] = t_code

    # Langkah B: Proses seluruh 59 Set Pieces secara kronologis
    set_pieces.sort(key=lambda x: x[0])
    for s_frame, e_frame, t_code, tags in set_pieces:
        sp_type = (tags.get("Type") or tags.get("type") or "").strip().lower()

        # Cari frame eksak saat bola ditendang/dilempar kembali ke permainan (r_frame)
        r_frame = s_frame
        search_end = min(n_frames - 1, max(s_frame + 1, min(e_frame, s_frame + 40)))
        for f in range(s_frame, search_end + 1):
            bx, by = ball_x_raw[f], ball_y_raw[f]
            bspd = ball_speed[f]
            in_pitch = (
                not np.isnan(bx)
                and not np.isnan(by)
                and (0.0 <= bx <= length)
                and (0.0 <= by <= width)
            )
            if in_pitch and (not np.isnan(bspd) and bspd >= 2.8):
                r_frame = f
                break

        # Inferensi tim pelempar/penendang jika Team = N/A (khususnya seluruh 26 Throw-in)
        sp_team = t_code
        if sp_team == NOBODY:
            post_slice = prior_team[r_frame : min(n_frames, r_frame + 75)]
            h_votes = int(np.sum(post_slice == HOME))
            a_votes = int(np.sum(post_slice == AWAY))
            if h_votes > a_votes + 5:
                sp_team = HOME
            elif a_votes > h_votes + 5:
                sp_team = AWAY
            else:
                w_s = max(0, r_frame - 12)
                w_e = min(n_frames, r_frame + 45)
                dh_win = dist_h1[w_s:w_e]
                da_win = dist_a1[w_s:w_e]
                h_near = int(np.sum(~np.isnan(dh_win) & (dh_win < np.nan_to_num(da_win, nan=999.0))))
                a_near = int(np.sum(~np.isnan(da_win) & (da_win < np.nan_to_num(dh_win, nan=999.0))))
                sp_team = HOME if h_near >= a_near else AWAY

        # Proteksi fase hidup sesudah restart (mencegah 'D' bocor ke bola lambung hasil tendangan)
        alive_end = min(n_frames, r_frame + 65)
        alive_restart_window[r_frame:alive_end] = True
        if sp_team != NOBODY:
            prior_team[r_frame : min(n_frames, r_frame + 60)] = sp_team

        # Jangan tandai dead ball di dalam babak 1 untuk Kick Off pembuka babak 1 (frame ~11185)
        if abs(s_frame - 11185) < 200 or abs(s_frame - SECOND_HALF_PRE_KICKOFF[1]) < 200:
            continue

        # Tentukan awal jeda bola mati (d_start) sebelum r_frame secara presisi
        base_pre = 80 if "throw" in sp_type else 115
        d_start = max(0, r_frame - base_pre)
        max_lookback = max(0, r_frame - 350)

        cur = d_start
        while cur > max_lookback:
            if alive_restart_window[cur - 1]:
                break
            bx, by = ball_x_raw[cur - 1], ball_y_raw[cur - 1]
            bspd = ball_speed[cur - 1]
            pspd = mean_player_speed[cur - 1]
            is_oob = (
                not np.isnan(bx)
                and ((bx < 0.15) or (bx > length - 0.15) or (by < 0.15) or (by > width - 0.15))
            )
            is_slow_ball = not np.isnan(bspd) and (bspd < 2.8)
            is_not_fast_ball = np.isnan(bspd) or (bspd < 3.6)
            is_slow_players = not np.isnan(pspd) and (pspd < 2.25)

            # Hanya mundur jika bola hilang, di luar garis, bola lambat, atau pemain berjalan saat bola tidak cepat
            if ball_missing_raw[cur - 1] or is_oob or is_slow_ball or (is_not_fast_ball and is_slow_players):
                cur -= 1
            elif gap_len[max(0, cur - 25) : cur].max(initial=0) >= 25 and is_not_fast_ball:
                cur -= 1
            else:
                break

        d_start = cur
        dead_ball_window[d_start:r_frame] = True
        alive_restart_window[d_start:r_frame] = False

    print(
        f"      Berhasil memetakan {len(events)} fase permainan & {len(set_pieces)} Set Pieces."
    )
    return prior_team, dead_ball_window, alive_restart_window


def build_decoupled_tracks(
    n_frames,
    pitch,
    ball_x_raw,
    ball_y_raw,
    ball_speed,
    ball_missing_raw,
    gap_len,
    gap_near_boundary,
    dist_h1,
    dist_h2,
    dist_a1,
    dist_a2,
    mean_player_speed,
    prior_team,
    dead_ball_window,
    alive_restart_window,
    dead_ball_speed=2.1,
    dead_player_speed=2.20,
):
    """
    Membangun jalur Dead Ball (raw_dead) dan jalur Tim (raw_team) tanpa histeresis lag buatan.
    dead_ball_speed / dead_player_speed: ambang (m/s) untuk melanjutkan status D saat bola
    nyaris diam dan pemain berjalan. Default = nilai v3; naikkan untuk menaikkan recall_D.
    """
    print("[3/5] Menghitung jalur Dead Ball (D) dan jalur Penguasaan Tim (H vs A)...")
    length, width = pitch
    raw_dead = np.zeros(n_frames, dtype=bool)
    raw_team = np.full(n_frames, "H", dtype="<U1")

    margin = 0.05
    out_of_bounds = (
        ~ball_missing_raw
        & (
            (ball_x_raw < -margin)
            | (ball_x_raw > length + margin)
            | (ball_y_raw < -margin)
            | (ball_y_raw > width + margin)
        )
    )

    for half, start_f, end_f in half_video_blocks(GAME):
        last_team = "A" if half == 1 else "H"
        in_dead_state = False

        for f in range(start_f, end_f + 1):
            # A. Evaluasi Status Dead Ball (raw_dead)
            if SECOND_HALF_PRE_KICKOFF[0] <= f <= SECOND_HALF_PRE_KICKOFF[1]:
                raw_dead[f] = True
                in_dead_state = True
            elif dead_ball_window[f]:
                raw_dead[f] = True
                in_dead_state = True
            elif alive_restart_window[f]:
                raw_dead[f] = False
                in_dead_state = False
            elif out_of_bounds[f]:
                raw_dead[f] = True
                in_dead_state = True
            elif ball_missing_raw[f]:
                g = gap_len[f]
                pspd = mean_player_speed[f]
                slow_players = not np.isnan(pspd) and (pspd < 2.28)
                if g >= 145 or (g >= 90 and (gap_near_boundary[f] or slow_players)):
                    raw_dead[f] = True
                    in_dead_state = True
                elif in_dead_state and g >= 35:
                    raw_dead[f] = True
                else:
                    raw_dead[f] = False
                    in_dead_state = False
            else:
                bspd = ball_speed[f]
                pspd = mean_player_speed[f]
                if (
                    in_dead_state
                    and (not np.isnan(bspd) and bspd < dead_ball_speed)
                    and (np.isnan(pspd) or pspd < dead_player_speed)
                ):
                    raw_dead[f] = True
                else:
                    raw_dead[f] = False
                    in_dead_state = False

            # B. Evaluasi Tim Penguasa Bola (raw_team: murni 'H' atau 'A', responsif tanpa lag)
            p_team = prior_team[f]
            if alive_restart_window[f] and p_team != NOBODY:
                last_team = "H" if p_team == HOME else "A"
                raw_team[f] = last_team
                continue

            dh1 = dist_h1[f]
            da1 = dist_a1[f]

            # Jika bola hilang > 35 frame (NaN), gunakan prior_team atau tahan last_team (BUKAN 'D')
            if np.isnan(dh1) and np.isnan(da1):
                if p_team == HOME:
                    last_team = "H"
                elif p_team == AWAY:
                    last_team = "A"
                raw_team[f] = last_team
                continue

            dh1_val = float(dh1) if not np.isnan(dh1) else 999.0
            da1_val = float(da1) if not np.isnan(da1) else 999.0

            # Saat kedua tim jauh dari bola (> 6.0m, umpan jauh), gunakan prior atau tahan last_team
            if min(dh1_val, da1_val) > 6.0:
                if p_team == HOME:
                    last_team = "H"
                elif p_team == AWAY:
                    last_team = "A"
                raw_team[f] = last_team
                continue

            dh2_val = float(dist_h2[f]) if not np.isnan(dist_h2[f]) else 12.0
            da2_val = float(dist_a2[f]) if not np.isnan(dist_a2[f]) else 12.0

            # Bandingkan jarak terdekat + dukungan tipis pemain ke-2 + bonus prior Smart Tagging (0.75m)
            eff_h = dh1_val + 0.06 * min(dh2_val, 10.0) - (0.75 if p_team == HOME else 0.0)
            eff_a = da1_val + 0.06 * min(da2_val, 10.0) - (0.75 if p_team == AWAY else 0.0)

            chosen = "H" if eff_h <= eff_a else "A"
            raw_team[f] = chosen
            last_team = chosen

    return raw_dead, raw_team, out_of_bounds


def _get_runs(arr):
    """Mengembalikan daftar segmen beruntun [[label, start_idx, end_idx], ...]."""
    runs = []
    if len(arr) == 0:
        return runs
    cur_lbl = arr[0]
    cur_s = 0
    for i in range(1, len(arr)):
        if arr[i] != cur_lbl:
            runs.append([cur_lbl, cur_s, i - 1])
            cur_lbl = arr[i]
            cur_s = i
    runs.append([cur_lbl, cur_s, len(arr) - 1])
    return runs


def _snap_team_transitions(combined, dist_h1_seg, dist_a1_seg, max_shift=10):
    """
    Transition Boundary Snapping:
    Menggeser titik perbatasan antara segmen H dan A (maksimal +/- max_shift frame)
    agar jatuh tepat pada frame saat jarak pemain ke bola (dist_h1 vs dist_a1) bersilangan.
    Tidak mengubah jumlah segmen (poss tetap), namun mempertajam trans_f1_12 dan IoU spellF1.
    """
    runs = _get_runs(combined)
    if len(runs) < 2:
        return combined

    n = len(combined)
    # Selisih jarak: negatif berarti Home lebih dekat, positif berarti Away lebih dekat
    diff = dist_h1_seg - dist_a1_seg

    for idx in range(len(runs) - 1):
        lbl_left, s_left, e_left = runs[idx]
        lbl_right, s_right, e_right = runs[idx + 1]
        if lbl_left not in ("H", "A") or lbl_right not in ("H", "A") or lbl_left == lbl_right:
            continue

        b = s_right  # Titik transisi saat ini (frame pertama segmen kanan)
        win_l = max(s_left + 20, b - max_shift)
        win_r = min(e_right - 20, b + max_shift)
        if win_l >= win_r:
            continue

        best_b = b
        best_score = -1e9
        for cand in range(win_l, win_r + 1):
            left_slice = diff[win_l:cand]
            right_slice = diff[cand : win_r + 1]
            valid_l = left_slice[~np.isnan(left_slice)]
            valid_r = right_slice[~np.isnan(right_slice)]

            # Untuk H -> A: sisi kiri harusnya diff < 0, sisi kanan diff > 0
            if lbl_left == "H":
                score_l = float(np.sum(valid_l < 0.0)) - float(np.sum(valid_l > 0.0))
                score_r = float(np.sum(valid_r > 0.0)) - float(np.sum(valid_r < 0.0))
            else:
                score_l = float(np.sum(valid_l > 0.0)) - float(np.sum(valid_l < 0.0))
                score_r = float(np.sum(valid_r < 0.0)) - float(np.sum(valid_r > 0.0))

            # Penalti sangat kecil untuk pergeseran jauh agar stabil jika tidak ada sinyal jelas
            score = score_l + score_r - 0.05 * abs(cand - b)
            if score > best_score:
                best_score = score
                best_b = cand

        if best_b < b:
            combined[best_b:b] = lbl_right
            runs[idx][2] = best_b - 1
            runs[idx + 1][1] = best_b
        elif best_b > b:
            combined[b:best_b] = lbl_left
            runs[idx][2] = best_b - 1
            runs[idx + 1][1] = best_b

    return combined


def _delay_team_transitions(combined, delay, min_keep=20):
    """
    Menggeser setiap batas H <-> A maju sebesar `delay` frame (segmen kiri diperpanjang).
    Mengoreksi bias 'terlalu awal' (mean_latency negatif) pada scoreboard v3. Batas yang
    bersebelahan dengan D tidak diubah, dan segmen kanan selalu menyisakan >= min_keep frame.
    """
    if delay <= 0:
        return combined
    runs = _get_runs(combined)  # dihitung sekali dari label asli
    out = combined.copy()
    for idx in range(len(runs) - 1):
        lbl_l, _, _ = runs[idx]
        lbl_r, s_r, e_r = runs[idx + 1]
        if lbl_l in ("H", "A") and lbl_r in ("H", "A") and lbl_l != lbl_r:
            shift = min(delay, max(0, (e_r - s_r + 1) - min_keep))
            if shift > 0:
                out[s_r : s_r + shift] = lbl_l
    return out


def _delay_restart_boundaries(combined, delay, min_keep=20):
    """
    Menggeser setiap batas D -> H/A maju sebesar `delay` frame (segmen D diperpanjang),
    mengoreksi bias 'terlalu awal' pada saat bola dilepas kembali ke permainan.
    Segmen tim di sebelah kanan selalu menyisakan >= min_keep frame. Batas H <-> A dan
    batas H/A -> D tidak diubah.
    """
    if delay <= 0:
        return combined
    runs = _get_runs(combined)  # dihitung sekali dari label asli
    out = combined.copy()
    for idx in range(len(runs) - 1):
        lbl_l, _, _ = runs[idx]
        lbl_r, s_r, e_r = runs[idx + 1]
        if lbl_l == "D" and lbl_r in ("H", "A"):
            shift = min(delay, max(0, (e_r - s_r + 1) - min_keep))
            if shift > 0:
                out[s_r : s_r + shift] = "D"
    return out


def smooth_and_segment_decoupled(
    raw_dead,
    raw_team,
    dead_ball_window,
    alive_restart_window,
    out_of_bounds,
    dist_h1,
    dist_a1,
    window_team=50,
    window_dead=35,
    min_team_spell=45,
    min_dead_spell=75,
    transition_delay=8,
    restart_delay=12,
):
    """
    Menjalankan smoothing terpisah (decoupled) untuk jalur tim (H vs A) dan jalur bola mati (D),
    diikuti oleh pembersihan segmen defleksi/boundary singkat dan Transition Boundary Snapping.
    """
    print(
        f"[4/5] Decoupled Smoothing (Team=+/-{window_team}, Dead=+/-{window_dead}) & Boundary Snapping..."
    )
    n_frames = len(raw_team)
    final_labels = np.full(n_frames, "X", dtype="<U1")

    for half, start_f, end_f in half_video_blocks(GAME):
        active_start = max(start_f, SECOND_HALF_PRE_KICKOFF[1] + 1) if half == 2 else start_f
        if half == 2:
            final_labels[SECOND_HALF_PRE_KICKOFF[0] : SECOND_HALF_PRE_KICKOFF[1] + 1] = "D"

        t_seg = raw_team[active_start : end_f + 1].copy()
        d_seg = raw_dead[active_start : end_f + 1].copy()
        sp_d_seg = dead_ball_window[active_start : end_f + 1]
        sp_alive_seg = alive_restart_window[active_start : end_f + 1]
        oob_seg = out_of_bounds[active_start : end_f + 1]
        dh1_seg = dist_h1[active_start : end_f + 1]
        da1_seg = dist_a1[active_start : end_f + 1]
        n = len(t_seg)

        # 1. Majority-vote smoothing untuk jalur Tim (H vs A)
        c_h = np.concatenate(([0], np.cumsum(t_seg == "H", dtype=np.int32)))
        c_a = np.concatenate(([0], np.cumsum(t_seg == "A", dtype=np.int32)))
        idx = np.arange(n)
        low_t = np.maximum(0, idx - window_team)
        high_t = np.minimum(n, idx + window_team + 1)
        team_smooth = np.where((c_h[high_t] - c_h[low_t]) >= (c_a[high_t] - c_a[low_t]), "H", "A").astype(
            "<U1"
        )

        # Bersihkan segmen H/A yang terlalu pendek pada jalur murni tim (< 45 frame / 1.8s)
        changed = True
        while changed:
            changed = False
            runs = _get_runs(team_smooth)
            for r_idx, (lbl, r_s, r_e) in enumerate(runs):
                if (r_e - r_s + 1) < min_team_spell and len(runs) > 1:
                    rep = runs[r_idx - 1][0] if r_idx > 0 else runs[r_idx + 1][0]
                    team_smooth[r_s : r_e + 1] = rep
                    changed = True
                    break

        # 2. Smoothing & morfologi untuk jalur Dead Ball (D)
        c_d = np.concatenate(([0], np.cumsum(d_seg, dtype=np.int32)))
        low_d = np.maximum(0, idx - window_dead)
        high_d = np.minimum(n, idx + window_dead + 1)
        win_len_d = high_d - low_d
        dead_smooth = (c_d[high_d] - c_d[low_d]) * 2 > win_len_d

        # Kunci jendela Set Piece yang sudah terkonfirmasi
        dead_smooth[sp_d_seg] = True
        dead_smooth[sp_alive_seg] = False

        # Jembatani celah alive yang sangat pendek (< 45 frame / 1.8 detik) di dalam satu jeda D
        d_runs = _get_runs(dead_smooth)
        for r_idx, (is_d, r_s, r_e) in enumerate(d_runs):
            if not is_d and (r_e - r_s + 1) < 45 and 0 < r_idx < len(d_runs) - 1:
                if not np.any(sp_alive_seg[r_s : r_e + 1]):
                    dead_smooth[r_s : r_e + 1] = True

        # Hapus segmen D yang lebih pendek dari min_dead_spell (3.0 detik)
        d_runs = _get_runs(dead_smooth)
        for is_d, r_s, r_e in d_runs:
            if is_d and (r_e - r_s + 1) < min_dead_spell:
                if not np.any(sp_d_seg[r_s : r_e + 1]):
                    dead_smooth[r_s : r_e + 1] = False

        # 3. Gabungkan jalur Dead Ball di atas jalur Tim
        combined = np.where(dead_smooth, "D", team_smooth).astype("<U1")

        # 4. Aturan Anti-Split: Jika ada segmen D pendek/menengah (< 170 frame) yang diapit oleh
        #    tim yang SAMA (H-D-H atau A-D-A) tanpa adanya Set Piece maupun bola keluar garis,
        #    kembalikan menjadi penguasaan tim tersebut agar tidak membelah penguasaan bola utuh.
        c_runs = _get_runs(combined)
        for r_idx, (lbl, r_s, r_e) in enumerate(c_runs):
            if lbl == "D" and 0 < r_idx < len(c_runs) - 1:
                prev_lbl = c_runs[r_idx - 1][0]
                next_lbl = c_runs[r_idx + 1][0]
                dur = r_e - r_s + 1
                has_sp = bool(np.any(sp_d_seg[r_s : r_e + 1]))
                oob_cnt = int(np.sum(oob_seg[r_s : r_e + 1]))
                if prev_lbl == next_lbl and prev_lbl in ("H", "A") and dur < 170 and not has_sp and oob_cnt < 10:
                    combined[r_s : r_e + 1] = prev_lbl

        # 5. Pembersihan Segmen Tim Pendek Pasca-Overlay
        #    - Sandwiched (H-A-H / A-H-A) < 52 frame (2.1s) diserap ke tim pengapit.
        #    - Boundary dekat D < 45 frame (1.8s) diserap ke tim tetangga (mempertahankan penguasaan >= 49 frame).
        changed = True
        while changed:
            changed = False
            c_runs = _get_runs(combined)
            for r_idx, (lbl, r_s, r_e) in enumerate(c_runs):
                if lbl not in ("H", "A"):
                    continue
                dur = r_e - r_s + 1
                prev_lbl = c_runs[r_idx - 1][0] if r_idx > 0 else None
                next_lbl = c_runs[r_idx + 1][0] if r_idx < len(c_runs) - 1 else None

                if prev_lbl == next_lbl and prev_lbl in ("H", "A") and dur < 52:
                    combined[r_s : r_e + 1] = prev_lbl
                    changed = True
                    break

                if dur < 45:
                    if prev_lbl == "D" and next_lbl in ("H", "A"):
                        combined[r_s : r_e + 1] = next_lbl
                        changed = True
                        break
                    if next_lbl == "D" and prev_lbl in ("H", "A"):
                        combined[r_s : r_e + 1] = prev_lbl
                        changed = True
                        break
                    if prev_lbl in ("H", "A"):
                        combined[r_s : r_e + 1] = prev_lbl
                        changed = True
                        break
                    if next_lbl in ("H", "A"):
                        combined[r_s : r_e + 1] = next_lbl
                        changed = True
                        break

        # 6. Transition Boundary Snapping (+/- 10 frame) untuk mempertajam batas pergantian H <-> A
        combined = _snap_team_transitions(combined, dh1_seg, da1_seg, max_shift=10)

        # 7. Koreksi bias latency: geser batas H <-> A maju (v3 terlalu awal ~10 frame)
        combined = _delay_team_transitions(combined, transition_delay)

        # 8. Koreksi bias latency pada batas D -> H/A (bola dilepas kembali)
        combined = _delay_restart_boundaries(combined, restart_delay)

        final_labels[active_start : end_f + 1] = combined

    return final_labels


def main():
    parser = argparse.ArgumentParser(
        description="More Than A Hack 2027 — Precision-Tuned Possession Pipeline (v7)"
    )
    parser.add_argument(
        "--team", type=str, default="Alamak Menang Wak", help="Nama tim untuk file CSV"
    )
    parser.add_argument(
        "--window-team",
        type=int,
        default=50,
        help="Half-width smoothing window untuk H vs A (frame); 60 = perilaku v6",
    )
    parser.add_argument(
        "--window-dead", type=int, default=35, help="Half-width smoothing window untuk D (frame)"
    )
    parser.add_argument(
        "--min-spell", type=int, default=45, help="Durasi minimum segmen penguasaan bola (frame)"
    )
    parser.add_argument(
        "--min-dead", type=int, default=75, help="Durasi minimum segmen bola mati / stoppage (frame)"
    )
    parser.add_argument(
        "--transition-delay",
        type=int,
        default=8,
        help="Geser batas H<->A maju (frame) untuk mengoreksi bias latency negatif; 0 = nonaktif",
    )
    parser.add_argument(
        "--restart-delay",
        type=int,
        default=12,
        help="Geser batas D->H/A maju (frame) saat bola dilepas kembali; 8 = v5, 0 = v4",
    )
    parser.add_argument(
        "--dead-ball-speed", type=float, default=2.1, help="Ambang kecepatan bola (m/s) lanjutan status D"
    )
    parser.add_argument(
        "--dead-player-speed",
        type=float,
        default=2.20,
        help="Ambang kecepatan rata-rata pemain (m/s) lanjutan status D",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Folder data opsional")
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, args.data_dir)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]

    home_name, away_name = team_names(meta_path)
    print(f"Tim Pertandingan: H = {home_name}, A = {away_name} | Total Frame = {n_frames}")

    team_of_track = team_of_each_track(meta_path)
    (
        ball_x_raw,
        ball_y_raw,
        ball_x,
        ball_y,
        ball_speed,
        ball_missing_raw,
        gap_len,
        gap_near_boundary,
        dist_h1,
        dist_h2,
        dist_a1,
        dist_a2,
        mean_player_speed,
    ) = stream_atd_two_pass(raw_path, team_of_track, n_frames, pitch)

    prior_team, dead_ball_window, alive_restart_window = extract_smart_tagging_context(
        n_frames,
        pitch,
        ball_x_raw,
        ball_y_raw,
        ball_speed,
        ball_missing_raw,
        gap_len,
        mean_player_speed,
        dist_h1,
        dist_a1,
        GAME,
        args.data_dir,
    )

    raw_dead, raw_team, out_of_bounds = build_decoupled_tracks(
        n_frames,
        pitch,
        ball_x_raw,
        ball_y_raw,
        ball_speed,
        ball_missing_raw,
        gap_len,
        gap_near_boundary,
        dist_h1,
        dist_h2,
        dist_a1,
        dist_a2,
        mean_player_speed,
        prior_team,
        dead_ball_window,
        alive_restart_window,
        dead_ball_speed=args.dead_ball_speed,
        dead_player_speed=args.dead_player_speed,
    )

    final_labels = smooth_and_segment_decoupled(
        raw_dead,
        raw_team,
        dead_ball_window,
        alive_restart_window,
        out_of_bounds,
        dist_h1,
        dist_a1,
        window_team=args.window_team,
        window_dead=args.window_dead,
        min_team_spell=args.min_spell,
        min_dead_spell=args.min_dead,
        transition_delay=args.transition_delay,
        restart_delay=args.restart_delay,
    )

    print("[5/5] Menyimpan file submission...")
    eval_out = os.path.join(ROOT, "evaluation", "results", f"{args.team}.csv")
    root_out = os.path.join(ROOT, f"{args.team}.csv")
    write_submission(eval_out, final_labels)
    shutil.copyfile(eval_out, root_out)

    counts = {k: int(np.sum(final_labels == k)) for k in ("H", "A", "D", "X")}
    graded_total = counts["H"] + counts["A"] + counts["D"]
    print(
        f"      Distribusi Label -> H: {counts['H']} ({100*counts['H']/graded_total:.1f}% graded), "
        f"A: {counts['A']} ({100*counts['A']/graded_total:.1f}% graded), "
        f"D: {counts['D']} ({100*counts['D']/graded_total:.1f}% graded), "
        f"X: {counts['X']}"
    )
    print(f"      Tersimpan di     -> {eval_out}")
    print(f"      Salinan upload   -> {root_out}")
    print("\nLangkah berikutnya:")
    print(f'  1. Validasi : python evaluation/validate_submission.py "{args.team}.csv"')
    print("  2. Cek skor : python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()