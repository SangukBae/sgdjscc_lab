"""tests/test_csi_estimation.py – pilot-free CSI estimation (paper Sec. IV-C).

CPU tests: SNR estimator (paper-like), phase estimator + joint loop (scaffold),
the self-supervised training pair, and the estimation losses. No GPU/checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.models.csi_estimation import (
    SNREstimator, PhaseEstimator, AFModule, joint_csi_estimate, build_csi_estimators,
)
from sgdjscc_lab.training.losses import (
    SNREstimationLoss, PhaseEstimationLoss, synthesize_noisy_latent,
)


def test_snr_estimator_outputs_unit_range():
    # raw net outputs a [0,1] scalar; its √α-vs-α meaning is set by the train target
    net = SNREstimator(latent_ch=16).eval()
    a = net(torch.randn(2, 16, 16, 16))
    assert a.shape == (2, 1)
    assert torch.all((a >= 0) & (a <= 1))


def test_af_module_preserves_shape():
    af = AFModule(channels=8)
    x = torch.randn(2, 8, 4, 4)
    out = af(x, torch.rand(2, 1))
    assert out.shape == x.shape


def test_phase_estimator_outputs_normalized_phase():
    net = PhaseEstimator(latent_ch=16).eval()
    phi = net(torch.randn(2, 16, 16, 16), torch.rand(2, 1))
    assert phi.shape == (2, 1)
    assert torch.all((phi >= -1) & (phi <= 1))         # φ/π ∈ [-1,1]


def test_joint_estimate_snr_only_and_with_phase():
    snr, phase = build_csi_estimators(latent_ch=16, with_phase=True)
    snr.eval(); phase.eval()
    feat = torch.randn(2, 16, 16, 16)
    # SNR-only (real-gain default): phi is zeros (scaffold no-op)
    a0, p0 = joint_csi_estimate(feat, snr, phase, complex_phase=False)
    assert a0.shape == (2, 1) and torch.count_nonzero(p0) == 0
    # complex_phase=True exercises the alternating loop (scaffold)
    a1, p1 = joint_csi_estimate(feat, snr, phase, max_iter=2, complex_phase=True)
    assert a1.shape == (2, 1) and p1.shape == (2, 1)
    # phase_estimator=None → SNR only
    a2, p2 = joint_csi_estimate(feat, snr, None)
    assert torch.count_nonzero(p2) == 0


def test_joint_estimate_converts_amplitude_output_to_alpha():
    # joint_csi_estimate must return the signal LEVEL α, converting the estimator's
    # default amplitude (√α) output via α = out² — so reusing the stage-default
    # estimator in the joint/phase scaffold keeps consistent semantics.
    net = SNREstimator(latent_ch=16).eval()
    assert net.output_is_amplitude is True
    feat = torch.randn(3, 16, 16, 16)
    out = net(feat).clamp(0, 1)
    alpha, _ = joint_csi_estimate(feat, net)                  # amplitude → α = out²
    assert torch.allclose(alpha, out ** 2, atol=1e-6)
    # explicit override: treat the estimator output as α directly
    alpha2, _ = joint_csi_estimate(feat, net, snr_is_amplitude=False)
    assert torch.allclose(alpha2, out, atol=1e-6)
    # a net flagged as α-output (output_is_amplitude=False) is used directly
    net.output_is_amplitude = False
    alpha3, _ = joint_csi_estimate(feat, net)
    assert torch.allclose(alpha3, out, atol=1e-6)


def test_loaded_estimator_carries_output_convention(tmp_path):
    from omegaconf import OmegaConf
    from sgdjscc_lab.training.stage_runners import CSIEstimationStageRunner
    from sgdjscc_lab.models.csi_estimation import load_snr_estimator

    def _save(target, name):
        est = SNREstimator(16)
        r = CSIEstimationStageRunner(
            lambda i: torch.randn(i.shape[0], 16, 16, 16), est,
            OmegaConf.create({"train": {"csi_estimation": {"target": target}}}),
            torch.device("cpu"), [{"params": list(est.parameters()), "name": "x"}])
        p = tmp_path / name
        torch.save({"runner_state": r.get_train_state()}, p)
        return p

    assert load_snr_estimator(str(_save("amplitude", "a.pth"))).output_is_amplitude is True
    assert load_snr_estimator(str(_save("alpha", "b.pth"))).output_is_amplitude is False


def test_synthesize_noisy_latent_endpoints():
    f0 = torch.randn(3, 16, 8, 8)
    g = torch.Generator().manual_seed(0)
    near1 = synthesize_noisy_latent(f0, torch.ones(3, 1), generator=g)
    assert torch.allclose(near1, f0, atol=1e-5)        # α=1 → clean
    g2 = torch.Generator().manual_seed(0)
    near0 = synthesize_noisy_latent(f0, torch.zeros(3, 1), generator=g2)
    assert not torch.allclose(near0, f0, atol=1e-1)    # α=0 → pure noise


def test_csi_losses_shapes_and_backprop():
    sl = SNREstimationLoss()
    a_hat = torch.rand(4, 1, requires_grad=True)
    out = sl(a_hat, torch.rand(4, 1))
    assert "loss_snr" in out and out["loss"].requires_grad
    out["loss"].backward()
    pl = PhaseEstimationLoss()
    p_hat = torch.rand(4, 1, requires_grad=True) * 2 - 1
    o2 = pl(p_hat, torch.rand(4, 1) * 2 - 1)
    assert "loss_phase" in o2 and o2["loss"].requires_grad


def test_load_snr_estimator_into_replaces_inference_predictor(tmp_path):
    # The trained estimator must actually drive the inference blind SNR path:
    # load it into jscc.snr_prediction_net and verify the runtime contract
    # (net(x).reshape([-1,1])**2 → signal level).
    from types import SimpleNamespace
    from sgdjscc_lab.models.csi_estimation import load_snr_estimator_into, load_snr_estimator

    est = SNREstimator(latent_ch=16)
    ckpt = tmp_path / "best.pth"
    torch.save({"epoch": 1, "stage": "csi_estimation",
                "runner_state": {"modules": {"snr_estimator": est.state_dict()}}}, ckpt)

    jscc = SimpleNamespace(snr_prediction_net=object())   # stand-in public net
    load_snr_estimator_into(jscc, str(ckpt), latent_ch=16)
    assert isinstance(jscc.snr_prediction_net, SNREstimator)
    # weights match the saved estimator
    for k, v in est.state_dict().items():
        assert torch.allclose(jscc.snr_prediction_net.state_dict()[k], v)
    # runtime contract: [B,16,h,w] → [B,1]; squaring gives a signal level in [0,1]
    x = torch.randn(2, 16, 16, 16)
    sig = jscc.snr_prediction_net(x).reshape([-1, 1]) ** 2
    assert sig.shape == (2, 1) and torch.all((sig >= 0) & (sig <= 1))
    # bare loader returns an eval-mode estimator
    assert load_snr_estimator(str(ckpt)).training is False


def test_alpha_target_checkpoint_is_sqrt_wrapped_on_load(tmp_path):
    # An α-target checkpoint must NOT be loaded raw into the net²=α runtime; the
    # loader reads the recorded target and √-wraps it so net² still yields α.
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from sgdjscc_lab.training.stage_runners import CSIEstimationStageRunner
    from sgdjscc_lab.models.csi_estimation import load_snr_estimator_into, _SqrtSNRAdapter

    est = SNREstimator(latent_ch=16)
    runner = CSIEstimationStageRunner(
        lambda i: torch.randn(i.shape[0], 16, 16, 16), est,
        OmegaConf.create({"train": {"csi_estimation": {"target": "alpha"}}}),
        torch.device("cpu"), [{"params": list(est.parameters()), "name": "x"}])
    ckpt = tmp_path / "alpha.pth"
    torch.save({"runner_state": runner.get_train_state()}, ckpt)   # records csi_target=alpha

    jscc = SimpleNamespace(snr_prediction_net=object())
    load_snr_estimator_into(jscc, str(ckpt), latent_ch=16)
    assert isinstance(jscc.snr_prediction_net, _SqrtSNRAdapter)     # wrapped, not raw
    x = torch.randn(2, 16, 16, 16)
    # net²(x) == inner α̂(x): squaring the wrapped output recovers the α the net predicts
    assert torch.allclose(jscc.snr_prediction_net(x) ** 2,
                          jscc.snr_prediction_net.inner(x).clamp(0, 1), atol=1e-6)


def test_amplitude_target_checkpoint_loads_raw(tmp_path):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from sgdjscc_lab.training.stage_runners import CSIEstimationStageRunner
    from sgdjscc_lab.models.csi_estimation import load_snr_estimator_into

    est = SNREstimator(latent_ch=16)
    runner = CSIEstimationStageRunner(
        lambda i: torch.randn(i.shape[0], 16, 16, 16), est,
        OmegaConf.create({"train": {"csi_estimation": {}}}),      # default amplitude
        torch.device("cpu"), [{"params": list(est.parameters()), "name": "x"}])
    ckpt = tmp_path / "amp.pth"
    torch.save({"runner_state": runner.get_train_state()}, ckpt)
    jscc = SimpleNamespace(snr_prediction_net=object())
    load_snr_estimator_into(jscc, str(ckpt), latent_ch=16)
    assert isinstance(jscc.snr_prediction_net, SNREstimator)        # raw drop-in


def test_csi_runner_target_amplitude_vs_alpha():
    from omegaconf import OmegaConf
    from sgdjscc_lab.training.stage_runners import CSIEstimationStageRunner
    enc = lambda imgs: torch.randn(imgs.shape[0], 16, 16, 16)
    batch = {"image": torch.rand(4, 3, 128, 128)}
    # default = amplitude (√α); inference squares the net output → α
    r_amp = CSIEstimationStageRunner(
        enc, SNREstimator(16), OmegaConf.create({"train": {"csi_estimation": {}}}),
        torch.device("cpu"), [{"params": [], "name": "x"}])
    assert r_amp.target == "amplitude"
    assert "loss_snr" in r_amp.forward(batch)
    # explicit alpha target (paper eq. 15 literal)
    r_a = CSIEstimationStageRunner(
        enc, SNREstimator(16),
        OmegaConf.create({"train": {"csi_estimation": {"target": "alpha"}}}),
        torch.device("cpu"), [{"params": [], "name": "x"}])
    assert r_a.target == "alpha"


def test_snr_estimator_is_learnable_on_synthetic_pairs():
    # 2-step smoke: the SNR estimator can reduce its loss toward its target.
    # Default contract is output_is_amplitude=True, so the regression target is
    # the signal amplitude √α (inference squares the output back to α).
    torch.manual_seed(0)
    net = SNREstimator(latent_ch=16).train()
    assert net.output_is_amplitude                     # default amplitude contract
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = SNREstimationLoss()
    f0 = torch.randn(8, 16, 16, 16)
    alpha = torch.rand(8, 1)
    target = alpha.sqrt()                              # √α, matching the net's output
    noisy = synthesize_noisy_latent(f0, alpha)
    first = None
    for _ in range(3):
        opt.zero_grad()
        out = loss_fn(net(noisy), target)
        out["loss"].backward()
        opt.step()
        if first is None:
            first = float(out["loss"])
    assert float(out["loss"]) <= first + 1e-6          # not diverging; gradient path OK
