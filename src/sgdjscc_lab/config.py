"""config.py – OmegaConf-based config loader with CLI override support.

Composition via ``_defaults_``
--------------------------------
A config file may declare a ``_defaults_`` list::

    _defaults_:
      - channel/awgn
      - model/sgdjscc
      - infer/awgn

Each entry is a path relative to the root config file's directory (without the
``.yaml`` extension).  ``load_config()`` loads the fragments in order, merges
them with ``OmegaConf.merge()``, then merges the root config on top so that
explicit keys in the root file always win. After composition all relative path
fields are resolved relative to the root config's directory. Workspace-aware
configs use the ``${sgdjscc:<kind>}`` resolver from :mod:`sgdjscc_lab.paths`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig, ListConfig

from .paths import lab_repo_root, register_omegaconf_resolver

register_omegaconf_resolver()

_PATH_KEYS = (
    "input_path",
    "output_dir",
    "model_root",
    "reference_path",
    "annotation_path",
    "caption_path",
    # Output artefacts – resolved relative to the config dir so results land in a
    # stable location regardless of the working directory the CLI is run from.
    "csv_path",
    "packet_dir",
    "keyframe_json",
    "temporal_csv",
    "frame_log_csv",
    "segment_json",
    # Training artefacts
    "train_input_path",
    "val_input_path",
    "checkpoint_dir",
    "train_log_path",
    # Trained blind SNR estimator (csi_estimation stage) loaded at inference
    "snr_estimator_checkpoint",
)


# Nested path fields (dotted) resolved the same way as the top-level ones, so
# they land relative to the config dir regardless of the CWD the CLI runs from.
_NESTED_PATH_KEYS = (
    # Video IO artefacts (evaluate_video.py)
    "video_io.extracted_frames_dir",           # mp4 → extracted frame folders
    "video_io.recon_frames_dir",               # reconstructed frame folder
    "video_io.recon_video",                    # re-assembled reconstruction mp4
    "verifier.report_json",                    # packet_match_report.json (ETRI 2차)
    "verifier.report_csv",                     # packet_match_report.csv (ETRI 2차)
    "verifier.decisions_json",                 # controller_decisions.json (ETRI 2차)
    "verifier.decisions_csv",                  # controller_decisions.csv (ETRI 2차)
    "video_generator.generated_frames_dir",    # generate-branch output frames (ETRI 3차)
    "video_generator.comparison_output",              # generation_mode_comparison.json (ETRI 4차)
    "video_generator.comparison_start_only_csv",      # start_only temporal metrics (ETRI 4차)
    "video_generator.comparison_bidirectional_csv",   # bidirectional temporal metrics (ETRI 4차)
    "video_generator.worker.python_bin",       # external worker's OTHER-env interpreter (ETRI 1B)
    "video_generator.worker.worker_script",    # lgvsc_generate_worker.py path override (ETRI 1B)
    "video_generator.worker.work_dir",         # per-call temp work dir base (ETRI 1B)
    "heldout.clip_only_json",                  # held-out remeasurement outputs (ETRI 5차)
    "heldout.clip_only_csv",
    "heldout.calibrated_json",
    "heldout.calibrated_csv",
    "heldout.output_json",
    "heldout.output_csv",
    "temporal_srs_calibration.weights_output",  # fitted SRS/temporal-SRS weights (ETRI 5차)
    # accounting.output_dir MUST resolve before the accounting.*_json/_csv keys
    # below (which interpolate ${accounting.output_dir}) — tuple order matters.
    "accounting.output_dir",                   # transmission accounting outputs (ETRI 6차)
    "accounting.frame_json",
    "accounting.frame_csv",
    "accounting.segment_json",
    "accounting.segment_csv",
    "accounting.summary_json",
    "rate_reliability.output_json",            # rate/reliability trade-off report (ETRI 6차)
    "rate_reliability.curve_csv",
    "train.controlnet.edge_jscc.checkpoint",   # trained edge codec for Stage 3
    "train.dataset.caption_path",              # manifest / coco_json / multi_manifest (train)
    "train.dataset.val_caption_path",          # manifest / coco_json / multi_manifest (val)
    "train.dataset.file_list_path",            # file-list mode (train images)
    "train.dataset.val_file_list_path",        # file-list mode (val images)
)


def _resolve_config_path(config_path: str | Path) -> Path:
    """Resolve a config path, including pre-reorganization legacy paths."""
    requested = Path(config_path).expanduser()
    if requested.exists():
        return requested.resolve()

    parts = requested.parts
    try:
        suffix = Path(*parts[parts.index("configs") + 1 :])
    except ValueError:
        return requested.resolve()

    root = lab_repo_root() / "configs"
    candidates = []
    if len(suffix.parts) > 1 and suffix.parts[0] in {
        "acceleration", "channel", "dataset", "eval", "infer", "model", "train", "video"
    }:
        candidates.append(root / "base" / suffix)
    if suffix == Path("default.yaml"):
        candidates.append(root / "base" / suffix)
    name = suffix.name
    if name.startswith("etri_video_eval"):
        candidates.append(root / "experiments" / "etri_video_eval" / name)
    if name.startswith("etri_lgvsc_1c"):
        candidates.append(root / "experiments" / "lgvsc_1c" / name)
    if name.startswith("paper_"):
        candidates.append(root / "experiments" / "paper_reproduction" / name)

    existing = [path for path in candidates if path.exists()]
    if len(existing) == 1:
        return existing[0].resolve()
    return requested.resolve()


def _resolve_paths(cfg: DictConfig, cfg_dir: Path) -> DictConfig:
    for key in _PATH_KEYS:
        val = cfg.get(key, None)
        if val is not None:
            p = Path(val)
            if not p.is_absolute():
                cfg[key] = str((cfg_dir / p).resolve())
    for dotted in _NESTED_PATH_KEYS:
        val = OmegaConf.select(cfg, dotted, default=None)
        if val is not None and not Path(val).is_absolute():
            OmegaConf.update(cfg, dotted, str((cfg_dir / Path(val)).resolve()),
                             force_add=False)
    return cfg


def load_config(config_path: str | Path) -> DictConfig:
    """Load YAML config from *config_path*, merging any ``_defaults_`` fragments.

    Paths inside the config are resolved relative to the **config file's
    directory**, not the current working directory.  This lets users run
    ``python scripts/infer_images.py --config configs/base/default.yaml`` from any
    working directory and still have ``input_path`` / ``output_dir`` /
    ``model_root`` resolve sensibly.

    If the config contains a ``_defaults_`` list, each entry names a fragment
    YAML (path without ``.yaml``, relative to ``cfg_dir``) that is loaded and
    merged in order before the root config is applied on top.
    """
    config_path = _resolve_config_path(config_path)
    cfg = OmegaConf.load(config_path)
    cfg_dir = config_path.parent

    # ── Fragment composition ──────────────────────────────────────────────────
    if "_defaults_" in cfg:
        defaults: list = OmegaConf.to_container(cfg.pop("_defaults_"), resolve=False)
        composed: DictConfig = OmegaConf.create({})
        for name in defaults:
            fragment_path = (cfg_dir / f"{name}.yaml").resolve()
            fragment = OmegaConf.load(fragment_path)
            composed = OmegaConf.merge(composed, fragment)
        # Root config keys override fragment defaults.
        cfg = OmegaConf.merge(composed, cfg)

    # ── Resolve relative path fields ──────────────────────────────────────────
    cfg = _resolve_paths(cfg, cfg_dir)

    return cfg


def merge_cli_overrides(
    cfg: DictConfig,
    input_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    snr_db: Optional[float] = None,
    device: Optional[str] = None,
) -> DictConfig:
    """Apply CLI flag overrides on top of a loaded config.

    Only non-None arguments override the corresponding config field.
    """
    overrides: dict = {}
    if input_path is not None:
        overrides["input_path"] = input_path
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if snr_db is not None:
        overrides["snr_db"] = snr_db
    if device is not None:
        overrides["device"] = device

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    return cfg
