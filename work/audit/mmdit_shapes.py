"""Phase 5 scoping: how many distinct static MMDiT graphs does the AR t2v path need?

CLAUDE.md records Phase 5 as "3 static graphs", from paper Table 7's three rows
[7x10x16], [7x20x32], [7x40x64]. That is a claim about resolution, not about
token count -- and the DiT's token count is what a static graph must pin.

This replays the pure shape arithmetic of the reference config with no weights,
no GPU and no torch: `_prepare_past_condition_latents` (generation_utils.py:178)
feeding `PatchEmbed3D.forward` (modeling_embedding.py:386), for every
(unit_index, stage) the AR loop visits.

Reference config, run_reference.py:192 and generation_utils.py:466-470:
    num_frames=49, num_stages=3, frames_per_unit=1, height=320, width=512
    -> latent 40x64, num_latent_frames = (49-1)//8 + 1 = 7, num_latent_units = 7
    -> an SSD1B first frame occupies unit 0, so the DiT runs units 1..6

  usage: py -3.10 work/audit/mmdit_shapes.py
"""

NUM_STAGES = 3
FRAMES_PER_UNIT = 1
NUM_LATENT_UNITS = 7
START_UNIT = 1              # unit 0 is the encoded SSD1B first frame
LAT_H, LAT_W = 40, 64       # 320/8, 512/8
PATCH = 2
TEXT_TOKENS = 128           # DistilT5 sequence length, see docs/device-results.md


def pyramid_hw(stage):
    """_get_pyramid_latent halves h,w per level and reverses: index 0 is coarsest."""
    div = 2 ** (NUM_STAGES - 1 - stage)
    return LAT_H // div, LAT_W // div


def past_condition_shapes(unit_index, stage):
    """Exactly generation_utils.py:178-227, in shapes only. Returns [(T,H,W), ...]."""
    if unit_index == 0:
        return []
    u = unit_index
    h, w = pyramid_hw(stage)
    stage_input = [(FRAMES_PER_UNIT, h, w)]          # last_cond_latent

    cur_unit_ptx = 1
    cur_stage = stage
    while cur_unit_ptx < u:
        cur_stage = max(cur_stage - 1, 0)
        if cur_stage == 0:
            break
        cur_unit_ptx += 1
        ch, cw = pyramid_hw(cur_stage)
        stage_input.append((FRAMES_PER_UNIT, ch, cw))

    if cur_stage == 0 and cur_unit_ptx < u:
        # history_latents_pyramid[0][:, :, : -(cur_unit_ptx * frames_per_unit)]
        t = u * FRAMES_PER_UNIT - cur_unit_ptx * FRAMES_PER_UNIT
        ch, cw = pyramid_hw(0)
        stage_input.append((t, ch, cw))

    return list(reversed(stage_input))


def tokens(shape):
    t, h, w = shape
    return t * (h // PATCH) * (w // PATCH)


def main():
    print("=" * 78)
    print("PHASE 5 SCOPING -- distinct MMDiT graph shapes on the AR t2v path")
    print("=" * 78)
    print("")
    print("  {:>4} {:>5} {:>7} {:>44} {:>8}".format(
        "unit", "stage", "tokens", "input latents [(T,H,W), ...]", "n_in"))
    print("  " + "-" * 72)

    seen = {}
    per_stage = {}
    for u in range(START_UNIT, NUM_LATENT_UNITS):
        for s in range(NUM_STAGES):
            h, w = pyramid_hw(s)
            seq = past_condition_shapes(u, s) + [(FRAMES_PER_UNIT, h, w)]
            n = sum(tokens(x) for x in seq)
            desc = " ".join("{}x{}x{}".format(*x) for x in seq)
            print("  {:>4} {:>5} {:>7} {:>44} {:>8}".format(u, s, n, desc, len(seq)))
            seen.setdefault((s, n), []).append(u)
            per_stage.setdefault(s, set()).add(n)
        print("")

    print("=" * 78)
    print("DISTINCT (stage, token count) combinations: {}".format(len(seen)))
    print("=" * 78)
    for (s, n), us in sorted(seen.items()):
        print("  stage {}  {:>5} image tokens (+{} text = {:>5} total)   units {}".format(
            s, n, TEXT_TOKENS, n + TEXT_TOKENS, us))

    print("")
    print("per-stage distinct token counts: " + ", ".join(
        "stage {}: {}".format(s, sorted(v)) for s, v in sorted(per_stage.items())))

    calls = (NUM_LATENT_UNITS - START_UNIT) * NUM_STAGES
    print("")
    print("DiT invocations per 49-frame video (1 step per stage): {}".format(calls))
    print("  paper Table 7 per-call: stage0 104.7 ms, stage1 218.3 ms, stage2 938.3 ms")
    print("  -> {} x (104.7 + 218.3 + 938.3) = {:.1f} ms of MMDiT per video".format(
        NUM_LATENT_UNITS - START_UNIT,
        (NUM_LATENT_UNITS - START_UNIT) * (104.7 + 218.3 + 938.3)))
    print("  (vs VAE enc 166.83 + dec 700 measured -> the MMDiT is the whole budget)")

    # what a padded-to-max design would cost
    print("")
    print("if every stage ran ONE graph padded to its own maximum:")
    for s, v in sorted(per_stage.items()):
        mx = max(v)
        waste = [(mx - n) / mx for n in sorted(v)]
        print("  stage {}: pad to {:>5} tokens, worst-case waste {:.0%}".format(
            s, mx, max(waste)))


if __name__ == "__main__":
    main()
