from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def mel_edges(
    bands: int,
    minimum_hz: float,
    maximum_hz: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    low = hz_to_mel(torch.tensor(float(minimum_hz), device=device, dtype=dtype))
    high = hz_to_mel(torch.tensor(float(maximum_hz), device=device, dtype=dtype))
    return mel_to_hz(torch.linspace(low, high, bands + 2, device=device, dtype=dtype))


def triangular_filterbank(frequencies: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    left = edges[:-2, None]
    centre = edges[1:-1, None]
    right = edges[2:, None]
    rising = (frequencies[None] - left) / (centre - left).clamp_min(1e-8)
    falling = (right - frequencies[None]) / (right - centre).clamp_min(1e-8)
    weights = torch.minimum(rising, falling).clamp_min(0.0)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)



def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _windowed_sinc_lowpass(taps: int, cutoff_cycles_per_sample: float) -> torch.Tensor:
    taps = int(taps)
    if taps < 3:
        raise ValueError("lowpass taps must be at least 3")
    if taps % 2 == 0:
        taps += 1
    cutoff = float(cutoff_cycles_per_sample)
    if not 0.0 < cutoff < 0.5:
        raise ValueError("cutoff_cycles_per_sample must lie in (0, 0.5)")
    n = torch.arange(taps, dtype=torch.float32) - (taps - 1) / 2.0
    kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * n)
    kernel = kernel * torch.blackman_window(taps, periodic=False)
    return kernel / kernel.sum().clamp_min(1e-12)



class FeatureFrontend(nn.Module):
    scale_channels: int
    num_scales: int

    @staticmethod
    def normalize_active(values, valid_mask, time_mask, eps: float = 1e-5):
        active = (valid_mask[:, :, None, None] & time_mask[:, None, None, :]).to(values.dtype)
        count = (active.sum(dim=(1, 2, 3), keepdim=True) * values.shape[2]).clamp_min(1.0)
        mean = (values * active).sum(dim=(1, 2, 3), keepdim=True) / count
        variance = ((values - mean).square() * active).sum(dim=(1, 2, 3), keepdim=True) / count
        return ((values - mean) / variance.sqrt().clamp_min(eps)) * active

    @staticmethod
    def normalize_per_channel(values, valid_mask, time_mask, eps: float = 1e-5):
        active = (valid_mask[:, :, None, None] & time_mask[:, None, None, :]).to(values.dtype)
        count = active.sum(dim=(1, 3), keepdim=True).clamp_min(1.0)
        mean = (values * active).sum(dim=(1, 3), keepdim=True) / count
        variance = ((values - mean).square() * active).sum(dim=(1, 3), keepdim=True) / count
        return ((values - mean) / variance.sqrt().clamp_min(eps)) * active

    @staticmethod
    def pad_scale_rows(rows):
        maximum_time = max(int(row.shape[-1]) for row in rows)
        padded, masks, counts = [], [], []
        for row in rows:
            frame_count = int(row.shape[-1])
            counts.append(frame_count)
            padded.append(F.pad(row, (0, maximum_time - frame_count)))
            mask = torch.zeros(maximum_time, dtype=torch.bool, device=row.device)
            mask[:frame_count] = True
            masks.append(mask)
        return torch.stack(padded), torch.stack(masks), torch.tensor(counts, dtype=torch.long, device=rows[0].device)


def resolve_window_and_hop(
    waveform_samples: int,
    sample_rate: int,
    duration_ms: float,
    hop_ratio: float,
    minimum_samples: int,
    maximum_samples: int,
) -> tuple[int, int]:
    if not 0.0 < float(hop_ratio) <= 1.0:
        raise ValueError("hop_ratio must be in (0, 1].")
    window_length = max(int(minimum_samples), int(round(sample_rate * duration_ms / 1000.0)))
    if int(maximum_samples) > 0:
        window_length = min(window_length, int(maximum_samples))
    window_length = min(max(4, window_length), max(4, int(waveform_samples)))
    hop_length = max(1, int(round(float(hop_ratio) * window_length)))
    return window_length, hop_length


class MelSpectrogramFrontend(FeatureFrontend):
    """Natural-length mel grids at one or more physical analysis durations."""

    def __init__(
        self,
        frequency_bins: int,
        minimum_hz: float,
        window_ms: tuple[float, ...],
        hop_ratio: float = 0.25,
        min_window_samples: int = 16,
        max_window_samples: int = 0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if not window_ms:
            raise ValueError("At least one mel analysis window is required.")
        self.frequency_bins = int(frequency_bins)
        self.minimum_hz = float(minimum_hz)
        self.window_ms = tuple(float(value) for value in window_ms)
        self.hop_ratio = float(hop_ratio)
        self.min_window_samples = int(min_window_samples)
        self.max_window_samples = int(max_window_samples)
        self.normalize = bool(normalize)
        self.scale_channels = 1
        self.num_scales = len(self.window_ms)

    def _one_window(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        duration_ms: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        window_length, hop_length = resolve_window_and_hop(
            waveform_samples=waveform.numel(),
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            hop_ratio=self.hop_ratio,
            minimum_samples=self.min_window_samples,
            maximum_samples=self.max_window_samples,
        )
        n_fft = next_power_of_two(window_length)
        if waveform.numel() < window_length:
            waveform = F.pad(waveform, (0, window_length - waveform.numel()))
        window = torch.hann_window(
            window_length, device=waveform.device, dtype=waveform.dtype
        )
        power = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_length,
            window=window,
            center=True,
            return_complex=True,
        ).abs().square()
        frequencies = torch.fft.rfftfreq(
            n_fft, d=1.0 / float(sample_rate), device=waveform.device
        ).to(waveform.dtype)
        edges = mel_edges(
            self.frequency_bins,
            self.minimum_hz,
            max(self.minimum_hz * 1.01, sample_rate / 2.0),
            waveform.device,
            waveform.dtype,
        )
        image = triangular_filterbank(frequencies, edges) @ power
        image = torch.log1p(image.clamp_min(0.0)).unsqueeze(1)  # [F,1,T]
        return (
            image,
            edges[1:-1],
            edges[2:] - edges[:-2],
            1000.0 * window_length / float(sample_rate),
            1000.0 * hop_length / float(sample_rate),
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rates: torch.Tensor,
        normalize: bool | None = None,
    ) -> dict[str, Any]:
        per_scale_rows: list[list[torch.Tensor]] = [list() for _ in self.window_ms]
        per_scale_window_ms: list[list[float]] = [list() for _ in self.window_ms]
        per_scale_hop_ms: list[list[float]] = [list() for _ in self.window_ms]
        centre_rows: list[torch.Tensor] = []
        bandwidth_rows: list[torch.Tensor] = []

        for index in range(len(waveforms)):
            waveform = waveforms[index, : int(lengths[index])].flatten().float()
            sample_rate = int(sample_rates[index])
            centres = bandwidths = None
            for scale_index, duration_ms in enumerate(self.window_ms):
                image, centres, bandwidths, realized_window, realized_hop = self._one_window(
                    waveform, sample_rate, duration_ms
                )
                per_scale_rows[scale_index].append(image)
                per_scale_window_ms[scale_index].append(realized_window)
                per_scale_hop_ms[scale_index].append(realized_hop)
            assert centres is not None and bandwidths is not None
            centre_rows.append(centres)
            bandwidth_rows.append(bandwidths)

        valid_mask = torch.ones(
            len(waveforms), self.frequency_bins, dtype=torch.bool, device=waveforms.device
        )
        apply_normalization = self.normalize if normalize is None else bool(normalize)
        scales: list[dict[str, torch.Tensor]] = []
        for scale_index, requested_window in enumerate(self.window_ms):
            raw, time_mask, frame_counts = self.pad_scale_rows(per_scale_rows[scale_index])
            features = (
                self.normalize_active(raw, valid_mask, time_mask)
                if apply_normalization
                else raw
            )
            scales.append(
                {
                    "features": features,
                    "raw_features": raw,
                    "time_mask": time_mask,
                    "frame_counts": frame_counts,
                    "requested_window_ms": torch.tensor(
                        float(requested_window), device=raw.device, dtype=raw.dtype
                    ),
                    "realized_window_ms": torch.tensor(
                        per_scale_window_ms[scale_index], device=raw.device, dtype=raw.dtype
                    ),
                    "realized_hop_ms": torch.tensor(
                        per_scale_hop_ms[scale_index], device=raw.device, dtype=raw.dtype
                    ),
                }
            )
        return {
            "scales": scales,
            "valid_mask": valid_mask,
            "centres_hz": torch.stack(centre_rows),
            "bandwidths_hz": torch.stack(bandwidth_rows),
        }



class NeuralAnalogResponse(nn.Module):
    """Continuous-frequency residual model around a log-Gaussian analog bank."""

    def __init__(
        self,
        num_filters: int,
        minimum_hz: float,
        maximum_hz: float,
        hidden_dim: int = 64,
        fourier_bands: int = 6,
        residual_scale: float = 1.5,
    ) -> None:
        super().__init__()
        self.num_filters = int(num_filters)
        self.minimum_hz = float(minimum_hz)
        self.maximum_hz = float(maximum_hz)
        self.fourier_bands = int(fourier_bands)
        self.residual_scale = float(residual_scale)
        input_dim = 1 + 2 * self.fourier_bands
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.num_filters),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def _features(self, frequencies_hz: torch.Tensor) -> torch.Tensor:
        safe = frequencies_hz.clamp_min(max(self.minimum_hz * 0.25, 1e-3))
        low = math.log(max(self.minimum_hz * 0.25, 1e-3))
        high = math.log(max(self.maximum_hz, self.minimum_hz * 1.01))
        u = 2.0 * (torch.log(safe) - low) / max(high - low, 1e-8) - 1.0
        features = [u]
        for index in range(self.fourier_bands):
            omega = math.pi * (2.0 ** index)
            features.append(torch.sin(omega * u))
            features.append(torch.cos(omega * u))
        return torch.stack(features, dim=-1)

    def forward(self, frequencies_hz: torch.Tensor) -> torch.Tensor:
        residual = self.network(self._features(frequencies_hz))
        return self.residual_scale * torch.tanh(residual)


class PhysicalPCEN(nn.Module):
    """Vectorized PCEN-like compression with a physical-time exponential smoother.

    The smoother is implemented as a short causal depthwise FIR approximation to an
    exponential low-pass. This removes the frame-by-frame Python recurrence from v7
    while retaining per-filter trainable PCEN parameters and a time constant in seconds.
    """

    def __init__(self, num_filters: int, smoother_frames: int = 64) -> None:
        super().__init__()
        num_filters = int(num_filters)
        self.smoother_frames = max(8, int(smoother_frames))
        self.alpha_logit = nn.Parameter(
            torch.full((num_filters,), _inverse_sigmoid(0.80), dtype=torch.float32)
        )
        self.delta_raw = nn.Parameter(torch.full((num_filters,), 1.0, dtype=torch.float32))
        self.root_logit = nn.Parameter(
            torch.full((num_filters,), _inverse_sigmoid((0.50 - 0.10) / 0.80), dtype=torch.float32)
        )
        self.tau_logit = nn.Parameter(
            torch.full((num_filters,), _inverse_sigmoid((0.060 - 0.010) / 0.110), dtype=torch.float32)
        )

    def forward(
        self,
        energy: torch.Tensor,
        frame_step_seconds: float,
        eps: float,
    ) -> torch.Tensor:
        if energy.ndim != 3:
            raise ValueError("PCEN expects [batch,frequency,time]")
        batch, filters, _ = energy.shape
        del batch
        alpha = 0.98 * torch.sigmoid(self.alpha_logit).to(energy)[None, :, None]
        delta = (F.softplus(self.delta_raw) + 1e-3).to(energy)[None, :, None]
        root = (0.10 + 0.80 * torch.sigmoid(self.root_logit)).to(energy)[None, :, None]
        tau = (0.010 + 0.110 * torch.sigmoid(self.tau_logit)).to(energy)

        dt = float(frame_step_seconds)
        lag = torch.arange(
            self.smoother_frames - 1,
            -1,
            -1,
            device=energy.device,
            dtype=energy.dtype,
        )[None, :]
        weights = torch.exp(-(lag * dt) / tau[:, None].clamp_min(1e-4))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
        padded = F.pad(
            energy,
            (self.smoother_frames - 1, 0),
            mode="replicate",
        )
        smooth = F.conv1d(
            padded,
            weights[:, None, :],
            groups=filters,
        )
        normalized = energy / (eps + smooth).pow(alpha)
        return (normalized + delta).pow(root) - delta.pow(root)


class SFIFrontend(FeatureFrontend):
    """Fast sampling-frequency-independent neural analog filterbank.

    Compared with v7, the mathematical representation is unchanged at the level used
    by the classifier: physical log-frequency analog responses are discretized for the
    native sampling rate, evaluated as real/quadrature filters, converted to envelopes,
    compressed, and summarized on 16/64/256 ms physical-time scales.

    The implementation is different where it matters for speed:
      * samples sharing a native rate are processed as one tensor;
      * the dyadic waveform pyramid is built once per rate group;
      * each rate/level kernel bank is generated once per forward, not once per item;
      * filter assignment is cached on the CPU with no per-filter GPU scalar reads;
      * no sinc realignment is performed in the hot path;
      * PCEN smoothing is vectorized and computed once before temporal pooling.
    """

    def __init__(
        self,
        minimum_hz: float,
        maximum_hz: float,
        frequency_bins: int = 128,
        q_factor: float = 12.0,
        temporal_windows_ms: tuple[float, ...] = (16.0, 64.0, 256.0),
        temporal_hops_ms: tuple[float, ...] = (4.0, 16.0, 64.0),
        base_frame_hop_ms: float = 4.0,
        nyquist_rolloff: float = 0.95,
        internal_oversampling: float = 4.0,
        minimum_internal_rate_hz: float = 256.0,
        filter_kernel_samples: int = 129,
        filter_design_fft: int = 512,
        anti_alias_taps: int = 31,
        naf_hidden_dim: int = 64,
        naf_fourier_bands: int = 6,
        naf_residual_scale: float = 1.5,
        pcen_smoother_frames: int = 64,
        normalize: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.minimum_hz = float(minimum_hz)
        self.maximum_hz = float(maximum_hz)
        self.frequency_bins = int(frequency_bins)
        self.q_factor = float(q_factor)
        self.temporal_windows_ms = tuple(float(v) for v in temporal_windows_ms)
        self.temporal_hops_ms = tuple(float(v) for v in temporal_hops_ms)
        self.base_frame_hop_ms = float(base_frame_hop_ms)
        self.nyquist_rolloff = float(nyquist_rolloff)
        self.internal_oversampling = float(internal_oversampling)
        self.minimum_internal_rate_hz = float(minimum_internal_rate_hz)
        self.filter_kernel_samples = int(filter_kernel_samples)
        if self.filter_kernel_samples % 2 == 0:
            self.filter_kernel_samples += 1
        self.filter_design_fft = max(
            next_power_of_two(int(filter_design_fft)),
            next_power_of_two(self.filter_kernel_samples * 2),
        )
        self.normalize = bool(normalize)
        self.eps = float(eps)
        if len(self.temporal_windows_ms) != len(self.temporal_hops_ms):
            raise ValueError("temporal_windows_ms and temporal_hops_ms must match")
        if len(self.temporal_windows_ms) != 3:
            raise ValueError("The temporal-pyramid backends expect three scales")
        if self.base_frame_hop_ms <= 0:
            raise ValueError("base_frame_hop_ms must be positive")
        if self.internal_oversampling <= 2.0:
            raise ValueError("internal_oversampling must be > 2")

        self.scale_channels = 2
        self.num_scales = len(self.temporal_windows_ms)

        centres = torch.logspace(
            math.log10(max(self.minimum_hz, 1e-3)),
            math.log10(max(self.maximum_hz, self.minimum_hz * 1.01)),
            self.frequency_bins,
        )
        sigma = max(1.0 / self.q_factor, 1e-4)
        half = math.sqrt(2.0 * math.log(2.0)) * sigma
        bandwidths = centres * (math.exp(half) - math.exp(-half))
        self.register_buffer("centres", centres, persistent=True)
        self.register_buffer("bandwidths", bandwidths, persistent=True)
        self.register_buffer(
            "anti_alias_kernel",
            _windowed_sinc_lowpass(anti_alias_taps, 0.225)[None, None, :],
            persistent=True,
        )
        self.register_buffer(
            "kernel_window",
            torch.hann_window(self.filter_kernel_samples, periodic=False),
            persistent=True,
        )

        self.analog_response = NeuralAnalogResponse(
            num_filters=self.frequency_bins,
            minimum_hz=self.minimum_hz,
            maximum_hz=self.maximum_hz,
            hidden_dim=int(naf_hidden_dim),
            fourier_bands=int(naf_fourier_bands),
            residual_scale=float(naf_residual_scale),
        )
        self.pcen = PhysicalPCEN(
            self.frequency_bins,
            smoother_frames=int(pcen_smoother_frames),
        )

        self._centres_python = [float(value) for value in centres.tolist()]
        self._geometry_cache: dict[int, tuple[list[bool], int, dict[int, list[int]]]] = {}

    def _geometry_for_rate(
        self,
        sample_rate: int,
    ) -> tuple[list[bool], int, dict[int, list[int]]]:
        sample_rate = int(sample_rate)
        cached = self._geometry_cache.get(sample_rate)
        if cached is not None:
            return cached

        valid = [
            centre <= 0.5 * sample_rate * self.nyquist_rolloff
            for centre in self._centres_python
        ]
        if not any(valid):
            valid[0] = True

        levels_for_filter: dict[int, list[int]] = {}
        deepest = 0
        for filter_index, (centre, is_valid) in enumerate(zip(self._centres_python, valid)):
            if not is_valid:
                continue
            required = max(
                self.minimum_internal_rate_hz,
                self.internal_oversampling * centre,
            )
            level = 0
            level_rate = float(sample_rate)
            while level_rate / 2.0 >= required:
                level_rate /= 2.0
                level += 1
            deepest = max(deepest, level)
            levels_for_filter.setdefault(level, []).append(filter_index)

        cached = (valid, deepest, levels_for_filter)
        self._geometry_cache[sample_rate] = cached
        return cached

    def _build_pyramid_batch(
        self,
        waveforms: torch.Tensor,
        deepest_level: int,
    ) -> list[torch.Tensor]:
        levels = [waveforms[:, None, :]]
        kernel = self.anti_alias_kernel.to(waveforms)
        for _ in range(int(deepest_level)):
            current = levels[-1]
            if current.shape[-1] <= 4:
                break
            reduced = F.conv1d(
                current,
                kernel,
                stride=2,
                padding=kernel.shape[-1] // 2,
            )
            levels.append(reduced)
        return levels

    def _continuous_response(
        self,
        frequencies_hz: torch.Tensor,
        filter_indices: torch.Tensor,
    ) -> torch.Tensor:
        centres = self.centres.to(frequencies_hz)[filter_indices]
        safe = frequencies_hz.clamp_min(max(self.minimum_hz * 0.25, 1e-3))
        log_ratio = torch.log(
            safe[None, :] / centres[:, None].clamp_min(self.eps)
        )
        sigma = max(1.0 / self.q_factor, 1e-4)
        base_log = -0.5 * (log_ratio / sigma).square()
        residual_all = self.analog_response(frequencies_hz)
        residual = residual_all[:, filter_indices].transpose(0, 1)
        response = torch.exp(base_log + residual)
        response = response * (frequencies_hz[None, :] > 0.0).to(response.dtype)
        return response / response.amax(dim=1, keepdim=True).clamp_min(self.eps)

    def _design_quadrature_kernels(
        self,
        sample_rate: float,
        filter_indices: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.fft.rfftfreq(
            self.filter_design_fft,
            d=1.0 / float(sample_rate),
            device=device,
        ).to(dtype)
        response = self._continuous_response(frequencies, filter_indices)
        real_impulse = torch.fft.irfft(response, n=self.filter_design_fft, dim=-1)
        imag_multiplier = torch.ones_like(response)
        imag_multiplier[:, 0] = 0.0
        if self.filter_design_fft % 2 == 0:
            imag_multiplier[:, -1] = 0.0
        quadrature_spectrum = torch.complex(
            torch.zeros_like(response), -response * imag_multiplier
        )
        imag_impulse = torch.fft.irfft(
            quadrature_spectrum,
            n=self.filter_design_fft,
            dim=-1,
        )
        real_impulse = torch.roll(real_impulse, self.filter_design_fft // 2, dims=-1)
        imag_impulse = torch.roll(imag_impulse, self.filter_design_fft // 2, dims=-1)
        start = (self.filter_design_fft - self.filter_kernel_samples) // 2
        stop = start + self.filter_kernel_samples
        window = self.kernel_window.to(device=device, dtype=dtype)[None, :]
        real_kernel = real_impulse[:, start:stop] * window
        imag_kernel = imag_impulse[:, start:stop] * window
        norm = torch.sqrt(
            real_kernel.square().sum(dim=1, keepdim=True)
            + imag_kernel.square().sum(dim=1, keepdim=True)
        ).clamp_min(self.eps)
        return real_kernel / norm, imag_kernel / norm

    @staticmethod
    def _crop_or_replicate_pad(values: torch.Tensor, target_frames: int) -> torch.Tensor:
        current = int(values.shape[-1])
        target_frames = int(target_frames)
        if current == target_frames:
            return values
        if current > target_frames:
            return values[..., :target_frames]
        if current == 0:
            return F.pad(values, (0, target_frames))
        return F.pad(values, (0, target_frames - current), mode="replicate")

    def _filter_level_batch(
        self,
        waveform_level: torch.Tensor,
        level_rate: float,
        filter_indices: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        real_kernel, imag_kernel = self._design_quadrature_kernels(
            level_rate,
            filter_indices,
            waveform_level.device,
            waveform_level.dtype,
        )
        kernels = torch.cat([real_kernel, imag_kernel], dim=0)[:, None, :]
        hop_samples = max(
            1,
            int(round(level_rate * self.base_frame_hop_ms / 1000.0)),
        )
        output = F.conv1d(
            waveform_level,
            kernels,
            stride=hop_samples,
            padding=self.filter_kernel_samples // 2,
        )
        count = int(filter_indices.numel())
        real = output[:, :count]
        imag = output[:, count:]
        envelope = torch.sqrt(real.square() + imag.square() + self.eps)
        return self._crop_or_replicate_pad(envelope, target_frames)

    @staticmethod
    def _temporal_pool(
        values: torch.Tensor,
        kernel_size: int,
        stride: int,
    ) -> torch.Tensor:
        kernel_size = max(1, int(kernel_size))
        stride = max(1, int(stride))
        if kernel_size == 1 and stride == 1:
            return values
        left = (kernel_size - 1) // 2
        right = kernel_size - 1 - left
        batch, frequency, channels, frames = values.shape
        merged = values.reshape(batch, frequency * channels, frames)
        merged = F.pad(merged, (left, right), mode="replicate")
        pooled = F.avg_pool1d(merged, kernel_size=kernel_size, stride=stride)
        return pooled.reshape(batch, frequency, channels, pooled.shape[-1])

    def _rate_group(
        self,
        waveforms: torch.Tensor,
        sample_rate: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        valid_list, deepest, assignments = self._geometry_for_rate(sample_rate)
        valid = torch.tensor(valid_list, dtype=torch.bool, device=waveforms.device)

        base_hop_samples = max(
            1,
            int(round(float(sample_rate) * self.base_frame_hop_ms / 1000.0)),
        )
        target_frames = max(1, int(waveforms.shape[-1] // base_hop_samples) + 1)
        levels = self._build_pyramid_batch(waveforms, deepest)

        envelope = waveforms.new_zeros(
            waveforms.shape[0],
            self.frequency_bins,
            target_frames,
        )
        for level, index_list in assignments.items():
            if level >= len(levels) or not index_list:
                continue
            filter_indices = torch.tensor(
                index_list,
                dtype=torch.long,
                device=waveforms.device,
            )
            level_rate = float(sample_rate) / float(2 ** level)
            filtered = self._filter_level_batch(
                levels[level],
                level_rate,
                filter_indices,
                target_frames,
            )
            envelope[:, filter_indices, :] = filtered

        envelope = envelope * valid[None, :, None].to(envelope.dtype)
        log_envelope = torch.log1p(envelope)
        pcen = self.pcen(
            envelope,
            frame_step_seconds=self.base_frame_hop_ms / 1000.0,
            eps=self.eps,
        )
        base = torch.stack([log_envelope, pcen], dim=2)
        base = base * valid[None, :, None, None].to(base.dtype)

        scales: list[torch.Tensor] = []
        for window_ms, hop_ms in zip(self.temporal_windows_ms, self.temporal_hops_ms):
            kernel_frames = max(1, int(round(window_ms / self.base_frame_hop_ms)))
            stride_frames = max(1, int(round(hop_ms / self.base_frame_hop_ms)))
            scales.append(self._temporal_pool(base, kernel_frames, stride_frames))
        return scales, valid

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rates: torch.Tensor,
        normalize: bool | None = None,
    ) -> dict[str, Any]:
        batch_size = int(waveforms.shape[0])
        rate_values = [int(v) for v in sample_rates.detach().cpu().tolist()]
        length_values = [int(v) for v in lengths.detach().cpu().tolist()]

        groups: dict[int, list[int]] = {}
        for index, rate in enumerate(rate_values):
            groups.setdefault(rate, []).append(index)

        per_scale_rows: list[list[torch.Tensor | None]] = [
            [None for _ in range(batch_size)] for _ in range(self.num_scales)
        ]
        mask_rows: list[torch.Tensor | None] = [None for _ in range(batch_size)]

        for rate, indices in groups.items():
            expected_length = length_values[indices[0]]
            if any(length_values[index] != expected_length for index in indices):
                raise RuntimeError("Items sharing a sample rate must share crop length.")
            index_tensor = torch.tensor(indices, dtype=torch.long, device=waveforms.device)
            group_waveforms = waveforms.index_select(0, index_tensor)[..., :expected_length]
            group_scales, valid = self._rate_group(group_waveforms, rate)
            for local_index, original_index in enumerate(indices):
                mask_rows[original_index] = valid
                for scale_index, scale in enumerate(group_scales):
                    per_scale_rows[scale_index][original_index] = scale[local_index]

        if any(mask is None for mask in mask_rows):
            raise RuntimeError("Internal SFI grouping error: missing mask row.")
        valid_mask = torch.stack([mask for mask in mask_rows if mask is not None], dim=0)
        apply_normalization = self.normalize if normalize is None else bool(normalize)

        scales: list[dict[str, torch.Tensor]] = []
        for scale_index, (window_ms, hop_ms) in enumerate(
            zip(self.temporal_windows_ms, self.temporal_hops_ms)
        ):
            rows = per_scale_rows[scale_index]
            if any(row is None for row in rows):
                raise RuntimeError("Internal SFI grouping error: missing scale row.")
            raw, time_mask, frame_counts = self.pad_scale_rows(
                [row for row in rows if row is not None]
            )
            features = (
                self.normalize_per_channel(raw, valid_mask, time_mask)
                if apply_normalization
                else raw
            )
            scales.append(
                {
                    "features": features,
                    "raw_features": raw,
                    "time_mask": time_mask,
                    "frame_counts": frame_counts,
                    "requested_window_ms": torch.tensor(
                        window_ms,
                        device=raw.device,
                        dtype=raw.dtype,
                    ),
                    "realized_window_ms": torch.full(
                        (batch_size,),
                        window_ms,
                        device=raw.device,
                        dtype=raw.dtype,
                    ),
                    "realized_hop_ms": torch.full(
                        (batch_size,),
                        hop_ms,
                        device=raw.device,
                        dtype=raw.dtype,
                    ),
                }
            )

        return {
            "scales": scales,
            "valid_mask": valid_mask,
            "centres_hz": self.centres.to(waveforms)[None].expand(batch_size, -1),
            "bandwidths_hz": self.bandwidths.to(waveforms)[None].expand(batch_size, -1),
        }

def group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(maximum), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock2d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: tuple[int, int] = (1, 1),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(group_count(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(output_channels), output_channels),
        )
        self.skip = nn.Identity()
        if input_channels != output_channels or stride != (1, 1):
            self.skip = nn.Conv2d(
                input_channels, output_channels, 1, stride=stride, bias=False
            )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.gelu(self.main(values) + self.skip(values)))


class StochasticDepth(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        probability = float(probability)
        if not 0.0 <= probability < 1.0:
            raise ValueError("stochastic-depth probability must be in [0, 1)")
        self.probability = probability

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return values
        survival = 1.0 - self.probability
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        keep = torch.empty(shape, device=values.device, dtype=values.dtype).bernoulli_(survival)
        return values * keep / survival


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, squeeze_channels, 1)
        self.expand = nn.Conv2d(squeeze_channels, channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scale = self.pool(values)
        scale = F.silu(self.reduce(scale))
        scale = torch.sigmoid(self.expand(scale))
        return values * scale


class MBConvBlock(nn.Module):
    """Mobile inverted bottleneck with depthwise convolution and SE gating."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        expansion_ratio: float,
        kernel_size: int,
        stride: int,
        se_ratio: float,
        stochastic_depth: float,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("MBConv kernel_size must be an odd integer >= 3")
        if stride not in (1, 2):
            raise ValueError("MBConv stride must be 1 or 2")
        expanded_channels = max(
            int(input_channels), int(round(input_channels * expansion_ratio))
        )
        layers: list[nn.Module] = []
        if expanded_channels != input_channels:
            layers.extend(
                [
                    nn.Conv2d(input_channels, expanded_channels, 1, bias=False),
                    nn.GroupNorm(group_count(expanded_channels), expanded_channels),
                    nn.SiLU(),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    expanded_channels,
                    expanded_channels,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=expanded_channels,
                    bias=False,
                ),
                nn.GroupNorm(group_count(expanded_channels), expanded_channels),
                nn.SiLU(),
                SqueezeExcitation(
                    expanded_channels,
                    max(8, int(round(input_channels * se_ratio))),
                ),
                nn.Conv2d(expanded_channels, output_channels, 1, bias=False),
                nn.GroupNorm(group_count(output_channels), output_channels),
            ]
        )
        self.main = nn.Sequential(*layers)
        self.use_residual = stride == 1 and input_channels == output_channels
        self.stochastic_depth = StochasticDepth(stochastic_depth)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.main(values)
        if self.use_residual:
            return values + self.stochastic_depth(update)
        return update


class SpectralConv2d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        modes_frequency: int,
        modes_time: int,
    ) -> None:
        super().__init__()
        self.output_channels = int(output_channels)
        self.modes_frequency = int(modes_frequency)
        self.modes_time = int(modes_time)
        scale = 1.0 / math.sqrt(max(1, input_channels * output_channels))
        shape = (
            input_channels,
            output_channels,
            self.modes_frequency,
            self.modes_time,
        )
        self.real_weight = nn.Parameter(scale * torch.randn(*shape))
        self.imag_weight = nn.Parameter(scale * torch.randn(*shape))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, _, frequency, time = values.shape
        spectrum = torch.fft.rfft2(values, norm="ortho")
        output = spectrum.new_zeros(
            batch, self.output_channels, frequency, time // 2 + 1
        )
        use_frequency = min(self.modes_frequency, frequency)
        use_time = min(self.modes_time, time // 2 + 1)
        weight = torch.complex(
            self.real_weight[:, :, :use_frequency, :use_time],
            self.imag_weight[:, :, :use_frequency, :use_time],
        )
        output[:, :, :use_frequency, :use_time] = torch.einsum(
            "bcft,coft->boft",
            spectrum[:, :, :use_frequency, :use_time],
            weight,
        )
        return torch.fft.irfft2(output, s=(frequency, time), norm="ortho")


class FNOBlock2d(nn.Module):
    def __init__(
        self,
        channels: int,
        modes_frequency: int,
        modes_time: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(
            channels, channels, modes_frequency, modes_time
        )
        self.local = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.spectral(values) + self.local(values)
        update = self.dropout(F.gelu(self.norm(update)))
        return values + update


def projection_block(input_channels: int, output_channels: int, activation: str) -> nn.Module:
    active: nn.Module = nn.SiLU() if activation == "silu" else nn.GELU()
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
        nn.GroupNorm(group_count(output_channels), output_channels),
        active,
    )


def transition_block(input_channels: int, output_channels: int, activation: str) -> nn.Module:
    active: nn.Module = nn.SiLU() if activation == "silu" else nn.GELU()
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 1, bias=False),
        nn.GroupNorm(group_count(output_channels), output_channels),
        active,
    )


class GridEncoderBase(nn.Module):
    output_dim: int

    def forward(
        self,
        scale_features: Sequence[torch.Tensor],
        valid_mask: torch.Tensor,
        time_masks: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TemporalPyramidEncoderBase(GridEncoderBase):
    """One encoder hierarchy with progressive natural-resolution scale fusion.

    The finest grid is processed first. Learned hidden features are average-pooled
    to the exact natural frame count of the next frontend scale, concatenated with
    a lightweight projection of that scale, and fused with a 1x1 convolution.
    Raw frontend grids are never temporally interpolated or stacked as channels.
    """

    def __init__(
        self,
        input_channels: int,
        num_scales: int,
        stage_channels: Sequence[int],
        embedding_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_scales = int(num_scales)
        self.stage_channels = tuple(int(value) for value in stage_channels)
        if self.num_scales not in (1, 3):
            raise ValueError("This experiment supports one or three frontend scales.")
        if len(self.stage_channels) < self.num_scales:
            raise ValueError("The encoder needs at least one stage per frontend scale.")
        self.output_dim = int(embedding_dim)
        self.dropout = float(dropout)
        self.activation_name = str(activation)

        self.input_projections = nn.ModuleList(
            [
                projection_block(
                    self.input_channels,
                    self.stage_channels[scale_index],
                    self.activation_name,
                )
                for scale_index in range(self.num_scales)
            ]
        )
        self.transitions = nn.ModuleList(
            [
                transition_block(
                    self.stage_channels[index],
                    self.stage_channels[index + 1],
                    self.activation_name,
                )
                for index in range(len(self.stage_channels) - 1)
            ]
        )
        self.fusions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        2 * self.stage_channels[index],
                        self.stage_channels[index],
                        1,
                        bias=False,
                    ),
                    nn.GroupNorm(
                        group_count(self.stage_channels[index]),
                        self.stage_channels[index],
                    ),
                    nn.SiLU() if self.activation_name == "silu" else nn.GELU(),
                )
                for index in range(1, self.num_scales)
            ]
        )
        self.stages = nn.ModuleList()
        self.stage_downsamples: list[bool] = []
        self.tail: nn.Module = nn.Identity()
        self.embedding_projection: nn.Module = nn.Identity()

    @staticmethod
    def safe_frequency_mask(
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        present = valid_mask.any(dim=1)
        safe_mask = valid_mask.clone()
        if (~present).any():
            safe_mask[~present, 0] = True
        return safe_mask, present

    @staticmethod
    def prepare_scale(
        features: torch.Tensor,
        safe_frequency_mask: torch.Tensor,
        time_mask: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_features = features
        if (~present).any():
            safe_features = features.clone()
            safe_features[~present] = 0.0
        spatial_mask = (
            safe_frequency_mask[:, None, :, None]
            & time_mask[:, None, None, :]
        ).to(features.dtype)
        inputs = safe_features.permute(0, 2, 1, 3).contiguous() * spatial_mask
        return inputs, spatial_mask

    @staticmethod
    def downsample_hidden_to_time(
        hidden: torch.Tensor,
        target_time: int,
    ) -> torch.Tensor:
        current_time = int(hidden.shape[-1])
        target_time = int(target_time)
        if target_time > current_time:
            raise ValueError(
                "Natural scale order is invalid: the next scale has more frames "
                f"({target_time}) than the current hidden map ({current_time})."
            )
        if target_time == current_time:
            return hidden
        return F.adaptive_avg_pool2d(hidden, (hidden.shape[-2], target_time))

    @staticmethod
    def halve_time(
        hidden: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape[-1] <= 1:
            return hidden, time_mask
        hidden = F.avg_pool2d(hidden, kernel_size=(1, 2), stride=(1, 2), ceil_mode=True)
        pooled_mask = F.max_pool1d(
            time_mask.to(hidden.dtype).unsqueeze(1),
            kernel_size=2,
            stride=2,
            ceil_mode=True,
        ).squeeze(1).bool()
        return hidden, pooled_mask

    @staticmethod
    def spatial_mask(
        frequency_mask: torch.Tensor,
        time_mask: torch.Tensor,
        frequency_size: int,
        time_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base = (
            frequency_mask[:, None, :, None]
            & time_mask[:, None, None, :]
        ).to(dtype)
        if base.shape[-2:] != (frequency_size, time_size):
            base = F.interpolate(base, size=(frequency_size, time_size), mode="nearest")
        return base

    @staticmethod
    def masked_global_pool(
        hidden: torch.Tensor,
        spatial_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = spatial_mask.to(hidden.dtype)
        count = mask.sum(dim=(-2, -1)).clamp_min(1.0)
        mean = (hidden * mask).sum(dim=(-2, -1)) / count
        masked = hidden.masked_fill(mask == 0, -torch.inf)
        maximum = masked.amax(dim=(-2, -1))
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return torch.cat([mean, maximum], dim=1)

    @staticmethod
    def row_weights(
        scale_features: Sequence[torch.Tensor],
        safe_frequency_mask: torch.Tensor,
        time_masks: Sequence[torch.Tensor],
        present: torch.Tensor,
    ) -> torch.Tensor:
        energies: list[torch.Tensor] = []
        for features, time_mask in zip(scale_features, time_masks):
            active = time_mask[:, None, None, :].to(features.dtype)
            count = active.sum(dim=(2, 3)).clamp_min(1.0) * features.shape[2]
            energy = (features.abs() * active).sum(dim=(2, 3)) / count
            energies.append(energy)
        mean_energy = torch.stack(energies, dim=0).mean(dim=0)
        mean_energy = mean_energy.masked_fill(~safe_frequency_mask, -torch.inf)
        weights = torch.softmax(mean_energy, dim=1)
        return weights * present.to(weights.dtype).unsqueeze(1)

    def forward(
        self,
        scale_features: Sequence[torch.Tensor],
        valid_mask: torch.Tensor,
        time_masks: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(scale_features) != self.num_scales or len(time_masks) != self.num_scales:
            raise ValueError(
                f"Encoder expects {self.num_scales} scales, received "
                f"{len(scale_features)} features and {len(time_masks)} masks."
            )
        safe_mask, present = self.safe_frequency_mask(valid_mask)
        prepared: list[torch.Tensor] = []
        input_masks: list[torch.Tensor] = []
        for features, time_mask in zip(scale_features, time_masks):
            inputs, spatial = self.prepare_scale(features, safe_mask, time_mask, present)
            prepared.append(inputs)
            input_masks.append(spatial)

        hidden = self.input_projections[0](prepared[0]) * input_masks[0]
        current_time_mask = time_masks[0]
        hidden = self.stages[0](hidden)
        current_spatial_mask = self.spatial_mask(
            safe_mask,
            current_time_mask,
            hidden.shape[-2],
            hidden.shape[-1],
            hidden.dtype,
        )
        hidden = hidden * current_spatial_mask

        for stage_index in range(1, len(self.stages)):
            hidden = self.transitions[stage_index - 1](hidden)
            if stage_index < self.num_scales:
                target_time = int(prepared[stage_index].shape[-1])
                hidden = self.downsample_hidden_to_time(hidden, target_time)
                current_time_mask = time_masks[stage_index]
                injected = self.input_projections[stage_index](prepared[stage_index])
                hidden = self.fusions[stage_index - 1](
                    torch.cat([hidden, injected], dim=1)
                )
            elif not self.stage_downsamples[stage_index]:
                hidden, current_time_mask = self.halve_time(hidden, current_time_mask)

            current_spatial_mask = self.spatial_mask(
                safe_mask,
                current_time_mask,
                hidden.shape[-2],
                hidden.shape[-1],
                hidden.dtype,
            )
            hidden = hidden * current_spatial_mask
            hidden = self.stages[stage_index](hidden)
            current_spatial_mask = self.spatial_mask(
                safe_mask,
                current_time_mask,
                hidden.shape[-2],
                hidden.shape[-1],
                hidden.dtype,
            )
            hidden = hidden * current_spatial_mask

        hidden = self.tail(hidden)
        final_mask = self.spatial_mask(
            safe_mask,
            current_time_mask,
            hidden.shape[-2],
            hidden.shape[-1],
            hidden.dtype,
        )
        hidden = hidden * final_mask
        pooled = self.masked_global_pool(hidden, final_mask)
        embedding = self.embedding_projection(pooled)
        embedding = embedding * present.to(embedding.dtype).unsqueeze(1)
        weights = self.row_weights(scale_features, safe_mask, time_masks, present)
        return embedding, weights


class CNNTemporalPyramidEncoder(TemporalPyramidEncoderBase):
    def __init__(
        self,
        input_channels: int,
        num_scales: int,
        base_channels: int,
        embedding_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        stage_channels = (int(base_channels), int(base_channels) * 2, int(base_channels) * 4)
        super().__init__(
            input_channels=input_channels,
            num_scales=num_scales,
            stage_channels=stage_channels,
            embedding_dim=embedding_dim,
            activation="gelu",
            dropout=dropout,
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ResidualBlock2d(channels, channels, dropout=dropout)
                        for _ in range(int(depth))
                    ]
                )
                for channels in stage_channels
            ]
        )
        self.stage_downsamples = [False] * len(self.stages)
        last = stage_channels[-1]
        self.tail = nn.Sequential(
            ResidualBlock2d(last, last * 2, (2, 2), dropout),
            ResidualBlock2d(last * 2, last * 4, (2, 2), dropout),
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(last * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )


class FNOTemporalPyramidEncoder(TemporalPyramidEncoderBase):
    def __init__(
        self,
        input_channels: int,
        num_scales: int,
        base_channels: int,
        embedding_dim: int,
        depth: int,
        modes_frequency: int,
        modes_time: int,
        dropout: float,
    ) -> None:
        stage_channels = (int(base_channels), int(base_channels) * 2, int(base_channels) * 4)
        super().__init__(
            input_channels=input_channels,
            num_scales=num_scales,
            stage_channels=stage_channels,
            embedding_dim=embedding_dim,
            activation="gelu",
            dropout=dropout,
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        FNOBlock2d(
                            channels,
                            modes_frequency=int(modes_frequency),
                            modes_time=int(modes_time),
                            dropout=dropout,
                        )
                        for _ in range(int(depth))
                    ]
                )
                for channels in stage_channels
            ]
        )
        self.stage_downsamples = [False] * len(self.stages)
        last = stage_channels[-1]
        self.tail = nn.Sequential(
            ResidualBlock2d(last, last * 2, (2, 2), dropout),
            ResidualBlock2d(last * 2, last * 4, (2, 2), dropout),
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(last * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )


class MBConvTemporalPyramidEncoder(TemporalPyramidEncoderBase):
    def __init__(
        self,
        input_channels: int,
        num_scales: int,
        embedding_dim: int,
        stage_channels: Sequence[int],
        stage_depths: Sequence[int],
        kernel_sizes: Sequence[int],
        expansion_ratio: float,
        se_ratio: float,
        stochastic_depth: float,
        head_channels: int,
        dropout: float,
    ) -> None:
        stage_channels = tuple(int(value) for value in stage_channels)
        stage_depths = tuple(int(value) for value in stage_depths)
        kernel_sizes = tuple(int(value) for value in kernel_sizes)
        if not (len(stage_channels) == len(stage_depths) == len(kernel_sizes)):
            raise ValueError(
                "MBConv stage_channels, stage_depths, and kernel_sizes must have equal lengths."
            )
        super().__init__(
            input_channels=input_channels,
            num_scales=num_scales,
            stage_channels=stage_channels,
            embedding_dim=embedding_dim,
            activation="silu",
            dropout=dropout,
        )
        total_blocks = sum(stage_depths)
        block_index = 0
        stages: list[nn.Module] = []
        downsample_flags: list[bool] = []
        for stage_index, (channels, repeats, kernel_size) in enumerate(
            zip(stage_channels, stage_depths, kernel_sizes)
        ):
            blocks: list[nn.Module] = []
            stage_downsample = stage_index >= 3
            for repeat_index in range(repeats):
                stride = 2 if stage_downsample and repeat_index == 0 else 1
                probability = float(stochastic_depth) * block_index / max(1, total_blocks - 1)
                blocks.append(
                    MBConvBlock(
                        input_channels=channels,
                        output_channels=channels,
                        expansion_ratio=float(expansion_ratio),
                        kernel_size=kernel_size,
                        stride=stride,
                        se_ratio=float(se_ratio),
                        stochastic_depth=probability,
                    )
                )
                block_index += 1
            stages.append(nn.Sequential(*blocks))
            downsample_flags.append(stage_downsample)
        self.stages = nn.ModuleList(stages)
        self.stage_downsamples = downsample_flags
        self.tail = nn.Sequential(
            nn.Conv2d(stage_channels[-1], int(head_channels), 1, bias=False),
            nn.GroupNorm(group_count(int(head_channels)), int(head_channels)),
            nn.SiLU(),
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(2 * int(head_channels), embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class ContrastiveProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.network(embedding), dim=1)


def frontend_metadata(front: dict) -> dict[str, torch.Tensor]:
    return {
        "centres_hz": front["centres_hz"],
        "bandwidths_hz": front["bandwidths_hz"],
        "valid_mask": front["valid_mask"],
        "scale_frame_counts": torch.stack(
            [scale["frame_counts"] for scale in front["scales"]], dim=1
        ),
        "scale_realized_window_ms": torch.stack(
            [scale["realized_window_ms"] for scale in front["scales"]], dim=1
        ),
        "scale_realized_hop_ms": torch.stack(
            [scale["realized_hop_ms"] for scale in front["scales"]], dim=1
        ),
    }


class ScaleFusionClassifier(nn.Module):
    """Frontend plus one natural-resolution temporal-pyramid encoder."""

    def __init__(
        self,
        frontend: FeatureFrontend,
        encoder: GridEncoderBase,
        num_classes: int,
        contrastive_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.output_dim, int(num_classes))
        self.contrastive_head = (
            ContrastiveProjection(encoder.output_dim, int(contrastive_dim))
            if contrastive_dim
            else None
        )
        self.embedding_dim = encoder.output_dim
        self.num_classes = int(num_classes)

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rates: torch.Tensor,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        front = self.frontend(waveforms, lengths, sample_rates, normalize=True)
        scale_features = [scale["features"] for scale in front["scales"]]
        time_masks = [scale["time_mask"] for scale in front["scales"]]
        embedding, weights = self.encoder(
            scale_features, front["valid_mask"], time_masks
        )
        logits = self.classifier(embedding)
        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "embedding": embedding,
            "row_weights": weights,
            **frontend_metadata(front),
        }
        if self.contrastive_head is not None:
            result["contrastive_embedding"] = self.contrastive_head(embedding)
        if return_features:
            for index, scale in enumerate(front["scales"]):
                result[f"features_scale_{index}"] = scale["features"]
                result[f"raw_features_scale_{index}"] = scale["raw_features"]
                result[f"time_mask_scale_{index}"] = scale["time_mask"]
        return result


def count_parameters(module: nn.Module) -> dict[str, int | float]:
    def count(child: nn.Module | None) -> int:
        if child is None:
            return 0
        return sum(parameter.numel() for parameter in child.parameters())

    total = count(module)
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    frontend = count(getattr(module, "frontend", None))
    encoder = count(getattr(module, "encoder", None))
    classifier = count(getattr(module, "classifier", None))
    contrastive = count(getattr(module, "contrastive_head", None))
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frontend_parameters": frontend,
        "encoder_parameters": encoder,
        "classifier_parameters": classifier,
        "contrastive_head_parameters": contrastive,
        "model_size_fp32_mb": total * 4.0 / 1024.0**2,
    }


MODEL_NAMES = [
    "mel48_cnn", "mel48_fno", "mel48_mbconv",
    "max_cnn", "max_fno", "max_mbconv",
    "sfi_cnn", "sfi_fno", "sfi_mbconv",
]

MODEL_SPECS = {
    "mel48_cnn": {"frontend": "mel48", "backend": "cnn", "rate_mode": "fixed_48k"},
    "mel48_fno": {"frontend": "mel48", "backend": "fno", "rate_mode": "fixed_48k"},
    "mel48_mbconv": {"frontend": "mel48", "backend": "mbconv", "rate_mode": "fixed_48k"},
    "max_cnn": {"frontend": "max", "backend": "cnn", "rate_mode": "corpus_max"},
    "max_fno": {"frontend": "max", "backend": "fno", "rate_mode": "corpus_max"},
    "max_mbconv": {"frontend": "max", "backend": "mbconv", "rate_mode": "corpus_max"},
    "sfi_cnn": {"frontend": "sfi", "backend": "cnn", "rate_mode": "native"},
    "sfi_fno": {"frontend": "sfi", "backend": "fno", "rate_mode": "native"},
    "sfi_mbconv": {"frontend": "sfi", "backend": "mbconv", "rate_mode": "native"},
}

DEFAULT_CONFIG = {
    "frequency_bins": 128,
    "minimum_frequency_hz": 5.0,
    "analysis_windows_ms": (16.0, 64.0, 256.0),
    "analysis_hops_ms": (4.0, 16.0, 64.0),
    "mel_hop_ratio": 0.25,
    "max_window_samples": 0,
    "q_factor": 12.0,
    "nyquist_rolloff": 0.95,
    "sfi_base_frame_hop_ms": 4.0,
    "sfi_internal_oversampling": 4.0,
    "sfi_minimum_internal_rate_hz": 256.0,
    "sfi_filter_kernel_samples": 129,
    "sfi_filter_design_fft": 512,
    "sfi_anti_alias_taps": 31,
    "sfi_hidden_dim": 64,
    "sfi_fourier_bands": 6,
    "sfi_residual_scale": 1.5,
    "sfi_pcen_smoother_frames": 64,
    "base_channels": 12,
    "embedding_dim": 192,
    "depth": 2,
    "fno_modes_frequency": 4,
    "fno_modes_time": 4,
    "mbconv_stage_channels": (24, 48, 96, 128),
    "mbconv_stage_depths": (2, 2, 2, 2),
    "mbconv_kernel_sizes": (3, 3, 5, 5),
    "mbconv_expansion_ratio": 4.0,
    "mbconv_se_ratio": 0.25,
    "mbconv_stochastic_depth": 0.10,
    "mbconv_head_channels": 192,
    "dropout": 0.10,
    "contrastive_projection_dim": 128,
}


def build_model(
    model_name: str,
    num_classes: int,
    maximum_frequency_hz: float,
    config: dict | None = None,
) -> nn.Module:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    spec = MODEL_SPECS[model_name]

    if spec["frontend"] in {"mel48", "max"}:
        frontend = MelSpectrogramFrontend(
            frequency_bins=cfg["frequency_bins"],
            minimum_hz=cfg["minimum_frequency_hz"],
            window_ms=tuple(cfg["analysis_windows_ms"]),
            hop_ratio=cfg["mel_hop_ratio"],
            max_window_samples=cfg["max_window_samples"],
            normalize=True,
        )
    else:
        frontend = SFIFrontend(
            minimum_hz=cfg["minimum_frequency_hz"],
            maximum_hz=float(maximum_frequency_hz),
            frequency_bins=cfg["frequency_bins"],
            q_factor=cfg["q_factor"],
            temporal_windows_ms=tuple(cfg["analysis_windows_ms"]),
            temporal_hops_ms=tuple(cfg["analysis_hops_ms"]),
            base_frame_hop_ms=cfg["sfi_base_frame_hop_ms"],
            nyquist_rolloff=cfg["nyquist_rolloff"],
            internal_oversampling=cfg["sfi_internal_oversampling"],
            minimum_internal_rate_hz=cfg["sfi_minimum_internal_rate_hz"],
            filter_kernel_samples=cfg["sfi_filter_kernel_samples"],
            filter_design_fft=cfg["sfi_filter_design_fft"],
            anti_alias_taps=cfg["sfi_anti_alias_taps"],
            naf_hidden_dim=cfg["sfi_hidden_dim"],
            naf_fourier_bands=cfg["sfi_fourier_bands"],
            naf_residual_scale=cfg["sfi_residual_scale"],
            pcen_smoother_frames=cfg["sfi_pcen_smoother_frames"],
            normalize=True,
        )

    backend = spec["backend"]
    if backend == "cnn":
        encoder = CNNTemporalPyramidEncoder(
            input_channels=frontend.scale_channels,
            num_scales=frontend.num_scales,
            base_channels=cfg["base_channels"],
            embedding_dim=cfg["embedding_dim"],
            depth=cfg["depth"],
            dropout=cfg["dropout"],
        )
    elif backend == "fno":
        encoder = FNOTemporalPyramidEncoder(
            input_channels=frontend.scale_channels,
            num_scales=frontend.num_scales,
            base_channels=cfg["base_channels"],
            embedding_dim=cfg["embedding_dim"],
            depth=cfg["depth"],
            modes_frequency=cfg["fno_modes_frequency"],
            modes_time=cfg["fno_modes_time"],
            dropout=cfg["dropout"],
        )
    else:
        encoder = MBConvTemporalPyramidEncoder(
            input_channels=frontend.scale_channels,
            num_scales=frontend.num_scales,
            embedding_dim=cfg["embedding_dim"],
            stage_channels=cfg["mbconv_stage_channels"],
            stage_depths=cfg["mbconv_stage_depths"],
            kernel_sizes=cfg["mbconv_kernel_sizes"],
            expansion_ratio=cfg["mbconv_expansion_ratio"],
            se_ratio=cfg["mbconv_se_ratio"],
            stochastic_depth=cfg["mbconv_stochastic_depth"],
            head_channels=cfg["mbconv_head_channels"],
            dropout=cfg["dropout"],
        )
    return ScaleFusionClassifier(
        frontend=frontend,
        encoder=encoder,
        num_classes=num_classes,
        contrastive_dim=cfg["contrastive_projection_dim"],
    )
