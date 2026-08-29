# Integrated semantic · hallucination · temporal evaluation

- status: `completed`
- coverage: 120/120 video-policy-profile pairs, 100 frames each
- fixed condition: fixed selector, int4 digital packet, fixed-reference SNR 10 dB, seed 2025
- baseline: `full50 + baseline`
- presence ensemble: CLIP + OWLv2 + VQA; evidence {'clip': 47744, 'owlv2': 47744, 'vqa': 47744}
- closed-world preservation: GT vocabulary filter
- open-world hallucination: non-object noise filter without GT vocabulary restriction
- selected development operating point: `few10 + candidate_both_omit`

The screening margins are provisional development gates, not a final claim.
Every effect is paired by video and includes a 95% bootstrap confidence interval
in `integrated_effect.csv`. A separate held-out validation remains required.
