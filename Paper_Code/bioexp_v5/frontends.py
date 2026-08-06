from __future__ import annotations

import math
from typing import Any

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


class FeatureFrontend(nn.Module):
    """Frontend interface returning natural-length grids for every analysis scale.

    Each element of ``scales`` contains:
      features:             [batch, frequency, channels, natural_time]
      raw_features:         same shape before normalization
      time_mask:            [batch, natural_time]
      frame_counts:         [batch]
      realized_window_ms:   [batch]
      realized_hop_ms:      [batch]
      requested_window_ms:  scalar tensor

    Shared fields:
      valid_mask:     [batch, frequency]
      centres_hz:     [batch, frequency]
      bandwidths_hz:  [batch, frequency]
    """

    scale_channels: int
    num_scales: int

    @staticmethod
    def normalize_active(
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        time_mask: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError("values must have shape [B,F,C,T]")
        frequency_active = valid_mask[:, :, None, None]
        time_active = time_mask[:, None, None, :]
        active = (frequency_active & time_active).to(values.dtype)
        channel_count = values.shape[2]
        count = active.sum(dim=(1, 2, 3), keepdim=True) * channel_count
        count = count.clamp_min(1.0)
        mean = (values * active).sum(dim=(1, 2, 3), keepdim=True) / count
        variance = ((values - mean).square() * active).sum(
            dim=(1, 2, 3), keepdim=True
        ) / count
        return ((values - mean) / variance.sqrt().clamp_min(eps)) * active

    @staticmethod
    def pad_scale_rows(
        rows: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not rows:
            raise ValueError("Cannot pad an empty scale.")
        maximum_time = max(int(row.shape[-1]) for row in rows)
        padded: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        counts: list[int] = []
        for row in rows:
            frame_count = int(row.shape[-1])
            counts.append(frame_count)
            padded.append(F.pad(row, (0, maximum_time - frame_count)))
            mask = torch.zeros(maximum_time, dtype=torch.bool, device=row.device)
            mask[:frame_count] = True
            masks.append(mask)
        return (
            torch.stack(padded, dim=0),
            torch.stack(masks, dim=0),
            torch.tensor(counts, dtype=torch.long, device=rows[0].device),
        )


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


class PhysicalModulationFrontend(FeatureFrontend):
    """Native-rate physical-frequency envelopes and modulation channels.

    The historical model names retain the ``sfi_`` prefix. This implementation
    is a native-rate physical-frequency modulation representation, not a claim
    of canonical SFI convolution.
    """

    def __init__(
        self,
        minimum_hz: float,
        maximum_hz: float,
        frequency_bins: int,
        q_factor: float,
        window_ms: tuple[float, ...],
        modulation_rates_hz: tuple[float, ...],
        nyquist_rolloff: float,
        hop_ratio: float = 0.25,
        min_window_samples: int = 16,
        max_window_samples: int = 0,
        normalize: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not window_ms:
            raise ValueError("At least one physical analysis window is required.")
        self.minimum_hz = float(minimum_hz)
        self.maximum_hz = float(maximum_hz)
        self.frequency_bins = int(frequency_bins)
        self.q_factor = float(q_factor)
        self.window_ms = tuple(float(value) for value in window_ms)
        self.modulation_rates_hz = tuple(float(value) for value in modulation_rates_hz)
        self.nyquist_rolloff = float(nyquist_rolloff)
        self.hop_ratio = float(hop_ratio)
        self.min_window_samples = int(min_window_samples)
        self.max_window_samples = int(max_window_samples)
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.scale_channels = 1 + len(self.modulation_rates_hz)
        self.num_scales = len(self.window_ms)

        centres = torch.logspace(
            math.log10(max(self.minimum_hz, 1e-3)),
            math.log10(self.maximum_hz),
            self.frequency_bins,
        )
        sigma = max(1.0 / self.q_factor, 1e-4)
        half = math.sqrt(2.0 * math.log(2.0)) * sigma
        bandwidths = centres * (math.exp(half) - math.exp(-half))
        self.register_buffer("centres", centres, persistent=True)
        self.register_buffer("bandwidths", bandwidths, persistent=True)

    def _stft_power(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        duration_ms: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
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
        return power, frequencies, window_length, hop_length

    def _physical_projection(
        self,
        power: torch.Tensor,
        source_frequencies: torch.Tensor,
        target_frequencies: torch.Tensor,
    ) -> torch.Tensor:
        safe_source = source_frequencies.clamp_min(self.minimum_hz * 0.25)
        log_ratio = torch.log(
            safe_source[None] / target_frequencies[:, None].clamp_min(self.eps)
        )
        sigma = max(1.0 / self.q_factor, 1e-3)
        weights = torch.exp(-0.5 * (log_ratio / sigma).square())
        weights *= (source_frequencies[None] > 0.0).to(weights.dtype)
        weights /= weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return weights @ power

    def _modulation_channels(
        self,
        envelope: torch.Tensor,
        frame_step_seconds: float,
    ) -> list[torch.Tensor]:
        channels = [envelope]
        if not self.modulation_rates_hz:
            return channels
        frames = envelope.shape[-1]
        centred = envelope - envelope.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(centred, dim=-1)
        axis = torch.fft.rfftfreq(
            frames,
            d=max(float(frame_step_seconds), self.eps),
            device=envelope.device,
        ).to(envelope.dtype)
        minimum_nonzero = 1.0 / max(frames * frame_step_seconds, self.eps)
        safe_axis = axis.clamp_min(minimum_nonzero)
        for rate in self.modulation_rates_hz:
            response = torch.exp(-0.5 * (torch.log(safe_axis / rate) / 0.50).square())
            response *= (axis > 0).to(response.dtype)
            modulation = torch.fft.irfft(
                spectrum * response[None].to(spectrum.dtype), n=frames, dim=-1
            ).abs()
            channels.append(torch.log1p(modulation))
        return channels

    def _one_scale(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        duration_ms: float,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        power, source, window_length, hop_length = self._stft_power(
            waveform, sample_rate, duration_ms
        )
        valid_target = self.centres.to(waveform)[valid]
        energy = self._physical_projection(power, source, valid_target)
        envelope = torch.log1p(torch.sqrt(energy.clamp_min(self.eps)))
        channels = self._modulation_channels(
            envelope, hop_length / float(sample_rate)
        )
        values = waveform.new_zeros(
            self.frequency_bins, self.scale_channels, envelope.shape[-1]
        )
        values[valid] = torch.stack(channels, dim=1)
        return (
            values,
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
        masks: list[torch.Tensor] = []

        for index in range(len(waveforms)):
            waveform = waveforms[index, : int(lengths[index])].flatten().float()
            if waveform.numel() < 8:
                waveform = F.pad(waveform, (0, 8 - waveform.numel()))
            sample_rate = int(sample_rates[index])
            target = self.centres.to(waveform)
            valid = target <= 0.5 * sample_rate * self.nyquist_rolloff
            if not valid.any():
                valid = valid.clone()
                valid[0] = True
            masks.append(valid)
            for scale_index, duration_ms in enumerate(self.window_ms):
                values, realized_window, realized_hop = self._one_scale(
                    waveform, sample_rate, duration_ms, valid
                )
                per_scale_rows[scale_index].append(values)
                per_scale_window_ms[scale_index].append(realized_window)
                per_scale_hop_ms[scale_index].append(realized_hop)

        valid_mask = torch.stack(masks).to(waveforms.device)
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
        batch = len(waveforms)
        return {
            "scales": scales,
            "valid_mask": valid_mask,
            "centres_hz": self.centres.to(waveforms)[None].expand(batch, -1),
            "bandwidths_hz": self.bandwidths.to(waveforms)[None].expand(batch, -1),
        }
