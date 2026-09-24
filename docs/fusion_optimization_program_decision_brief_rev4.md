# Fusion Optimization Program Decision Brief — Rev. 4

Date: 2026-09-24  
Decision owner: project/program owner; this brief prepares the choice and does not make it.

## Situation

The complete exact Tier-1 Fusion candidate now includes S1, S2, S3a, S4, S5, S6, and S7. On the
required composed non-identity workload it measures **204.257 ms/frame**, down **42.5%** from the
pre-S5/S7 candidate arm at **355.525 ms/frame**, but remains above the plan’s approximately 80 ms
practical ceiling. The raw scan/pose dataset is unavailable, so this is not an end-to-end result.

## Where time remains

Regime-B candidate profile:

- Rebin: 136.091 ms, 65.3%.
- Output conversion: 31.440 ms, 15.1%.
- Assembly/prune: 25.153 ms, 12.1%.
- Matching + dynamic detection: 13.317 ms combined, 6.4%.
- Measurement log-odds: 2.385 ms, 1.1%.

Rebin remains the dominant target. The stored map reaches 390,010 cells; average rebin unique-key
retention is 99.21%, with an average 2,743 collisions per nonempty frame.

## S21 viability signal

Non-identity key stability is highly ring-dependent: ring 0 = 0.0%, ring 1 = 0.0%, ring 2 = 1.6%,
ring 3 = 62.8% mean stable cells. No ring had an all-stable event on any eligible frame. This is
weak evidence for an all-or-nothing per-ring S21 fast path. It is evidence for the program decision,
not authorization to implement S21.

## Options for the owner

1. Prioritize serialization work. Historical snapshot serialization is 580.945 ms/frame, larger
   than historical Fusion at 248.778 ms/frame; reducing compression/copy cost may have greater
   end-to-end leverage.
2. Authorize further exact Fusion structural work, beginning with a measured S14/S8 investigation
   and only a narrowly conditioned S21 experiment if its invalidation rules can exploit partial
   stability without changing output order or float accumulation.
3. Obtain the raw 271-frame scan/pose dataset and run the end-to-end gate before selecting between
   the above options.

Historical context: Rating 208.123 ms/frame, model path 231.375 ms/frame, full recorded pipeline
1,059.836 ms/frame, combined sequential estimate 1,365.321 ms/frame. These figures are context,
not revised by this Fusion-only phase.

## Prepared conclusion

The exact Tier-1 path passes correctness but does not meet the regime-B ≤80 ms rule. The program
owner should choose between serialization levers and further exact Fusion structural work; no
choice is made by this brief.
