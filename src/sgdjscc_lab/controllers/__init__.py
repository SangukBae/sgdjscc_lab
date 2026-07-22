"""controllers/ – Phase 4 adaptive control policies.

Modules
-------
snr_guidance_policy        – SNR → guidance-parameter policy (regime table).
adaptive_guidance_controller – apply the SNR policy on top of a run config.
regeneration_policy        – error-type-aware regeneration strategy selection
                              (image path; applies concrete diffusion param patches).
verifier_controller         – error-type-aware decision + candidate-action logging
                              for the packet verifier (video/temporal path, ETRI
                              2차); decides but does not itself regenerate.
"""
