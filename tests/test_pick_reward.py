"""Hand-verified, physics-free check of so101_pick's staged reward math.

Feeds compute_pick_reward() exact, hand-picked numbers for a full attempt --
align -> open -> descend -> grasp -> lift -> success -- and checks every
term against a value worked out by hand from the formulas in
vision_rl/envs/so101_pick.py. No MJX/GPU/render needed, since the reward
math is pure (see compute_pick_reward's docstring for why it was pulled out
of step() into its own function).

Also regression-tests the two reward-hacking bugs found 2026-08-05/06 (see
so101_pick.py step() history): a flat per-step open bonus that let a policy
camp instead of descending, and a per-step descend delta that let a policy
farm reward by swinging up and down instead of committing to a real descent.

Run with:  python -m tests.test_pick_reward
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from vision_rl.config import SO101PickConfig
from vision_rl.envs.so101_pick import compute_pick_reward


def _r(cfg, d_xy, dz, cube_z, cube_init_z, grip_open, prev_d_xy, min_dz,
       was_open, was_picked, table_hit=0.0, ctrl_cost=0.0):
    """Thin wrapper: derive grip_closing from grip_open, like step() does.

    compute_pick_reward is written against jax arrays (uses .astype()), so
    plain Python floats need wrapping -- the real env always calls it with
    jnp arrays coming out of physics.
    """
    f = jnp.float32
    return compute_pick_reward(
        cfg, f(d_xy), f(dz), f(cube_z), f(cube_init_z),
        f(grip_open), f(1.0 - grip_open),
        f(prev_d_xy), f(min_dz), f(was_open), f(was_picked),
        f(table_hit), f(ctrl_cost),
    )


def _close(a, b, tol=1e-6):
    return math.isclose(float(a), float(b), abs_tol=tol)


def test_align_stage_far_from_cube():
    cfg = SO101PickConfig()
    out = _r(cfg, d_xy=0.08, dz=0.20, cube_z=0.0175, cube_init_z=0.0175,
              grip_open=0.0, prev_d_xy=0.10, min_dz=0.20,
              was_open=0.0, was_picked=0.0)
    # Not aligned (0.08 > align_tol=0.02): only align_r/align_pr fire.
    assert _close(out["align_r"], -1.0 * 0.08)           # -0.08
    assert _close(out["align_pr"], 3.0 * (0.10 - 0.08))  # +0.06
    assert _close(out["open_r"], 0.0)
    assert _close(out["descend_pr"], 0.0)
    assert _close(out["grasp_r"], 0.0)
    assert _close(out["lift_r"], 0.0)
    assert _close(out["succ_r"], 0.0)
    assert _close(out["reward"], -0.08 + 0.06)           # -0.02
    print("[ok] align stage: -0.08 (align_r) + 0.06 (align_pr) = -0.02")


def test_open_bonus_pays_once_not_every_step():
    cfg = SO101PickConfig()
    # Step 1: aligned, still above the cube (dz > grasp_z_tol), jaw opens.
    out1 = _r(cfg, d_xy=0.019, dz=0.20, cube_z=0.0175, cube_init_z=0.0175,
               grip_open=1.0, prev_d_xy=0.08, min_dz=0.20,
               was_open=0.0, was_picked=0.0)
    assert _close(out1["open_r"], 0.5)          # one-shot bonus fires
    assert _close(out1["was_open_next"], 1.0)   # latch flips on

    # Step 2: SAME pose held again (this is exactly the old bug's shape --
    # camping aligned+above+open). Latch is already 1, so it must pay 0.
    out2 = _r(cfg, d_xy=0.019, dz=0.20, cube_z=0.0175, cube_init_z=0.0175,
               grip_open=1.0, prev_d_xy=0.019, min_dz=0.20,
               was_open=out1["was_open_next"], was_picked=0.0)
    assert _close(out2["open_r"], 0.0)
    assert _close(out2["was_open_next"], 1.0)   # stays latched, doesn't reset
    print("[ok] open bonus: +0.5 once, then 0.0 forever after (BUG C fix)")


def test_descend_ratchet_pays_new_lows_only():
    cfg = SO101PickConfig()
    # First descent: 0.20 -> 0.15, a brand new low. Should pay in full.
    out = _r(cfg, d_xy=0.015, dz=0.15, cube_z=0.0175, cube_init_z=0.0175,
              grip_open=1.0, prev_d_xy=0.019, min_dz=0.20,
              was_open=1.0, was_picked=0.0)
    assert _close(out["descend_pr"], 3.0 * (0.20 - 0.15))  # +0.15
    assert _close(out["new_min_dz"], 0.15)
    print("[ok] descend ratchet: new low 0.20->0.15 pays +0.15")


def test_swing_exploit_no_longer_pays_on_revisit():
    """Regression test for BUG D (2026-08-06): reproduces the exact
    down-open / up-closed / down-open-again cycle observed in the trained
    checkpoint's rollout (dz sawtoothing between ~0.07 and ~0.25-0.33) and
    checks the second descent to an already-visited depth earns nothing.
    """
    cfg = SO101PickConfig()
    min_dz, was_open = 0.15, 1.0

    # D: descend to a new low, 0.15 -> 0.07, jaw open. Pays in full.
    d = _r(cfg, d_xy=0.012, dz=0.07, cube_z=0.0175, cube_init_z=0.0175,
           grip_open=1.0, prev_d_xy=0.015, min_dz=min_dz,
           was_open=was_open, was_picked=0.0)
    assert _close(d["descend_pr"], 3.0 * (0.15 - 0.07))  # +0.24
    min_dz = d["new_min_dz"]
    assert _close(min_dz, 0.07)

    # E: swing back up, 0.07 -> 0.25, jaw CLOSES (this is the exploit's
    # trick: closing during ascent used to dodge the matching debit).
    e = _r(cfg, d_xy=0.012, dz=0.25, cube_z=0.0175, cube_init_z=0.0175,
           grip_open=0.0, prev_d_xy=0.012, min_dz=min_dz,
           was_open=was_open, was_picked=0.0)
    assert _close(e["descend_pr"], 0.0)      # no debit -- expected, both old
                                              # and new formula gate on grip_open
    assert _close(e["new_min_dz"], 0.07)     # record does NOT reset upward
    min_dz = e["new_min_dz"]

    # F: descend again, back down to the SAME 0.07 -- already-visited ground.
    # OLD formula (descend_progress_scale * (prev_dz - dz)) would have paid
    # 3.0 * (0.25 - 0.07) = +0.54 here -- a full second payout for terrain
    # already covered in step D. That's the exploit.
    f = _r(cfg, d_xy=0.012, dz=0.07, cube_z=0.0175, cube_init_z=0.0175,
           grip_open=1.0, prev_d_xy=0.012, min_dz=min_dz,
           was_open=was_open, was_picked=0.0)
    old_formula_would_pay = 3.0 * (0.25 - 0.07)
    assert _close(old_formula_would_pay, 0.54)
    assert _close(f["descend_pr"], 0.0), (
        f"ratchet should block the revisit, got {f['descend_pr']}")

    cycle_total_new = d["descend_pr"] + e["descend_pr"] + f["descend_pr"]
    cycle_total_old_equiv = d["descend_pr"] + 0.0 + old_formula_would_pay
    assert _close(cycle_total_new, 0.24)
    assert _close(cycle_total_old_equiv, 0.78)
    print(f"[ok] swing exploit closed: one down-up-down cycle now nets "
          f"{cycle_total_new:.2f} instead of the old {cycle_total_old_equiv:.2f} "
          f"(the {cycle_total_old_equiv - cycle_total_new:.2f} gap is what a "
          f"policy used to be able to farm indefinitely by oscillating)")


def test_grasp_handoff_at_correct_position():
    cfg = SO101PickConfig()
    # Reaches grasp height (dz=0.025 < grasp_z_tol=0.03) exactly aligned,
    # jaw closes this same step.
    out = _r(cfg, d_xy=0.010, dz=0.025, cube_z=0.0175, cube_init_z=0.0175,
              grip_open=0.0, prev_d_xy=0.012, min_dz=0.07,
              was_open=1.0, was_picked=0.0)
    assert _close(out["held"], 1.0)
    assert _close(out["grasp_r"], 0.5)
    assert _close(out["descend_pr"], 0.0)   # jaw closed -> this term is off
    assert _close(out["success"], 0.0)      # cube hasn't risen yet
    print("[ok] grasp handoff: descend_pr -> 0, grasp_r -> +0.5, same step")


def test_grasp_at_wrong_position_earns_nothing_extra():
    cfg = SO101PickConfig()
    # Closes the jaw, but NOT aligned (d_xy=0.05 > align_tol) -- should not
    # earn grasp_r; this is what discourages closing prematurely/off-target.
    out = _r(cfg, d_xy=0.05, dz=0.025, cube_z=0.0175, cube_init_z=0.0175,
              grip_open=0.0, prev_d_xy=0.05, min_dz=0.07,
              was_open=1.0, was_picked=0.0)
    assert _close(out["held"], 0.0)
    assert _close(out["grasp_r"], 0.0)
    print("[ok] mistimed grasp (off-target) earns no grasp_r, as intended")


def test_approach_gate_shuts_off_reach_terms_once_lifted():
    cfg = SO101PickConfig()
    # Held (aligned + at height + closed), and the cube has genuinely risen
    # past lift_thresh (0.0525) -> approach_gate flips off. align_r/align_pr/
    # open_r/descend_pr must all go to exactly zero DESPITE d_xy/prev_d_xy
    # being nonzero (they'd read -0.01 and +0.12 respectively if the gate
    # weren't applied), while grasp_r/lift_r -- which don't depend on
    # approach_gate -- stay on.
    out = _r(cfg, d_xy=0.01, dz=0.02, cube_z=0.06, cube_init_z=0.0175,
              grip_open=0.0, prev_d_xy=0.05, min_dz=0.02,
              was_open=1.0, was_picked=0.0)
    assert _close(out["lifted_now"], 1.0)
    assert _close(out["held"], 1.0)
    assert _close(out["align_r"], 0.0)      # would be -0.01 if ungated
    assert _close(out["align_pr"], 0.0)     # would be +0.12 if ungated
    assert _close(out["open_r"], 0.0)
    assert _close(out["descend_pr"], 0.0)
    assert _close(out["grasp_r"], 0.5)                       # unaffected
    assert _close(out["lift_r"], 4.0 * (0.06 - 0.0175))      # unaffected
    print("[ok] approach_gate: reach-phase terms hard-zeroed once lifted, "
          "grasp_r/lift_r keep paying")


def test_lift_reward_scales_with_height_and_caps():
    cfg = SO101PickConfig()
    # lift_r caps once cube_lift (= cube_z - cube_init_z) reaches pick_height
    # (0.10) -- that's cube_z = cube_init_z + pick_height = 0.1175, NOT
    # cube_z = pick_height (pick_height is an absolute-cube_z threshold only
    # for the separate `success` check, which uses cube_z directly).
    low = _r(cfg, d_xy=0.01, dz=0.02, cube_z=0.04, cube_init_z=0.0175,
              grip_open=0.0, prev_d_xy=0.01, min_dz=0.02,
              was_open=1.0, was_picked=0.0)
    high = _r(cfg, d_xy=0.01, dz=0.02, cube_z=0.1175, cube_init_z=0.0175,
               grip_open=0.0, prev_d_xy=0.01, min_dz=0.02,
               was_open=1.0, was_picked=0.0)
    over_cap = _r(cfg, d_xy=0.01, dz=0.02, cube_z=0.20, cube_init_z=0.0175,
                   grip_open=0.0, prev_d_xy=0.01, min_dz=0.02,
                   was_open=1.0, was_picked=0.0)
    assert low["lift_r"] < high["lift_r"]                      # rises with height
    assert _close(high["lift_r"], over_cap["lift_r"])          # caps at pick_height
    assert _close(high["lift_r"], 4.0 * cfg.pick_height)        # exactly 0.40
    print("[ok] lift_r grows with height then caps at lift_scale*pick_height")


def test_success_bonus_only_once_height_and_hold_both_true():
    cfg = SO101PickConfig()
    out = _r(cfg, d_xy=0.010, dz=0.020, cube_z=0.12, cube_init_z=0.0175,
              grip_open=0.0, prev_d_xy=0.010, min_dz=0.02,
              was_open=1.0, was_picked=0.0)
    assert _close(out["success"], 1.0)
    assert _close(out["succ_r"], 5.0)
    assert _close(out["ever_picked"], 1.0)
    print("[ok] success_bonus fires once height AND hold both hold")


def test_ctrl_cost_and_table_penalty_subtract_correctly():
    cfg = SO101PickConfig()
    baseline = _r(cfg, d_xy=0.08, dz=0.20, cube_z=0.0175, cube_init_z=0.0175,
                   grip_open=0.0, prev_d_xy=0.08, min_dz=0.20,
                   was_open=0.0, was_picked=0.0, table_hit=0.0, ctrl_cost=0.0)
    penalized = _r(cfg, d_xy=0.08, dz=0.20, cube_z=0.0175, cube_init_z=0.0175,
                     grip_open=0.0, prev_d_xy=0.08, min_dz=0.20,
                     was_open=0.0, was_picked=0.0, table_hit=1.0, ctrl_cost=0.05)
    expected_drop = 0.05 + cfg.table_penalty_scale * 1.0   # ctrl_cost + table_pen
    assert _close(baseline["reward"] - penalized["reward"], expected_drop)
    print("[ok] ctrl_cost and table_pen subtract exactly, nothing else moves")


def test_full_attempt_end_to_end_totals():
    """Chains a full attempt -- align, open, descend, grasp, lift, success --
    threading each step's outputs into the next, exactly as step() does via
    the env's state fields. Every reward value below was independently
    computed by hand from the config constants; see the module docstring.
    """
    cfg = SO101PickConfig()
    prev_d_xy, min_dz, was_open, was_picked = 0.10, 0.20, 0.0, 0.0
    cube_z = 0.0175
    rewards = []

    steps = [
        # (d_xy,  dz,   grip_open, cube_z)
        (0.08,  0.20, 0.0, 0.0175),   # A: approaching, jaw closed
        (0.019, 0.20, 1.0, 0.0175),   # B: aligned, opens jaw (one-shot bonus)
        (0.015, 0.15, 1.0, 0.0175),   # C: descending, new low
        (0.012, 0.07, 1.0, 0.0175),   # D: descending further, new low
        (0.012, 0.25, 0.0, 0.0175),   # E: swings up, jaw closes (no debit)
        (0.012, 0.07, 1.0, 0.0175),   # F: swings back down -- OLD ground
        (0.010, 0.025, 0.0, 0.0175),  # G: at cube, closes -> grasp_r takes over
        (0.010, 0.020, 0.0, 0.06),    # H: lifting, approach_gate now off
        (0.010, 0.020, 0.0, 0.12),    # I: past pick_height -> success
    ]
    for d_xy, dz, grip_open, cz in steps:
        out = _r(cfg, d_xy, dz, cz, 0.0175, grip_open,
                  prev_d_xy, min_dz, was_open, was_picked)
        rewards.append(float(out["reward"]))
        prev_d_xy, min_dz = d_xy, float(out["new_min_dz"])
        was_open, was_picked = float(out["was_open_next"]), float(out["ever_picked"])

    expected = [
        -0.02,    # A: -0.08 + 0.06
        0.664,    # B: -0.019 + 0.183 + 0.5
        0.147,    # C: -0.015 + 0.012 + 0.15
        0.237,    # D: -0.012 + 0.009 + 0.24
        -0.012,   # E: -0.012 + 0 + 0 (swing up, no debit, no credit)
        -0.012,   # F: -0.012 + 0 + 0 (revisited ground -- ratchet blocks it)
        0.496,    # G: -0.010 + 0.006 + 0.5 (grasp handoff)
        0.67,     # H: 0 + 0 + 0.5 + 0.17 (approach_gate off, lift ramping)
        5.90,     # I: 0.5 + 0.40 (capped) + 5.0 (success)
    ]
    for i, (got, want) in enumerate(zip(rewards, expected)):
        assert _close(got, want, tol=1e-3), (
            f"step {i}: got reward {got:.4f}, expected {want:.4f}")
    print("[ok] full attempt end-to-end:")
    for i, r in enumerate(rewards):
        print(f"       step {i} ({'ABCDEFGHI'[i]}): reward = {r:+.3f}")
    print(f"       total over the attempt: {sum(rewards):+.3f}")


if __name__ == "__main__":
    test_align_stage_far_from_cube()
    test_open_bonus_pays_once_not_every_step()
    test_descend_ratchet_pays_new_lows_only()
    test_swing_exploit_no_longer_pays_on_revisit()
    test_grasp_handoff_at_correct_position()
    test_grasp_at_wrong_position_earns_nothing_extra()
    test_approach_gate_shuts_off_reach_terms_once_lifted()
    test_lift_reward_scales_with_height_and_caps()
    test_success_bonus_only_once_height_and_hold_both_true()
    test_ctrl_cost_and_table_penalty_subtract_correctly()
    test_full_attempt_end_to_end_totals()
    print("\nAll reward math checks passed.")
