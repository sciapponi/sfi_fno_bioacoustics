from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class MelFrontendConfig:
    frequency_bins: int = 96
    time_bins: int = 96
    minimum_hz: float = 50.0
    window_ms: tuple[float, ...] = (16.0, 64.0, 256.0)
    min_window_samples: int = 16
    max_window_samples: int = 8192
    normalize: bool = True


@dataclass(frozen=True)
class PhysicalModulationConfig:
    """Native-rate physical-frequency modulation frontend configuration.

    The historical model names retain the `sfi_` prefix, but this frontend is
    more precisely a native-rate, physical-frequency, modulation representation.
    """

    minimum_hz: float = 5.0
    maximum_hz: float = 120000.0
    frequency_bins: int = 96
    time_bins: int = 96
    q_factor: float = 12.0
    window_ms: tuple[float, ...] = (16.0, 64.0, 256.0)
    modulation_rates_hz: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
    nyquist_rolloff: float = 0.92
    min_window_samples: int = 16
    max_window_samples: int = 8192
    normalize: bool = True
    eps: float = 1e-6


class FeatureFrontend(nn.Module):
    """Common interface for every frontend in the experiment.

    Returned tensors:
      features:       [batch, frequency, channels, time]
      raw_features:   same shape before normalization
      valid_mask:     [batch, frequency]
      centres_hz:     [batch, frequency]
      bandwidths_hz:  [batch, frequency]
    """

    @property
    def output_channels(self) -> int:
        raise NotImplementedError

    @staticmethod
    def normalize_active(
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        active = valid_mask[:, :, None, None].to(values.dtype)
        count = active.sum(dim=(1, 2, 3), keepdim=True)
        count = count * values.shape[2] * values.shape[3]
        count = count.clamp_min(1.0)
        mean = (values * active).sum(dim=(1, 2, 3), keepdim=True) / count
        variance = ((values - mean).square() * active).sum(
            dim=(1, 2, 3), keepdim=True
        ) / count
        return ((values - mean) / variance.sqrt().clamp_min(eps)) * active


class MelSpectrogramFrontend(FeatureFrontend):
    """One or more mel spectrograms stacked as feature channels.

    Waveforms are resampled by the dataset before entering this frontend. Thus,
    all mel models use the same dataset-specific target-rate mapping.
    """

    def __init__(self, config: MelFrontendConfig) -> None:
        super().__init__()
        self.config = config
        if not config.window_ms:
            raise ValueError("At least one mel window is required.")

    @property
    def output_channels(self) -> int:
        return len(self.config.window_ms)

    def _one_window(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        duration_ms: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window_length = int(round(sample_rate * duration_ms / 1000.0))
        window_length = max(self.config.min_window_samples, window_length)
        window_length = min(self.config.max_window_samples, window_length)
        window_length = min(max(4, window_length), max(4, waveform.numel()))
        hop_length = max(1, int(round(0.25 * window_length)))
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
            self.config.frequency_bins,
            self.config.minimum_hz,
            max(self.config.minimum_hz * 1.01, sample_rate / 2.0),
            waveform.device,
            waveform.dtype,
        )
        image = triangular_filterbank(frequencies, edges) @ power
        image = torch.log1p(image.clamp_min(0.0))
        image = F.interpolate(
            image[None, None],
            size=(self.config.frequency_bins, self.config.time_bins),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return image, edges[1:-1], edges[2:] - edges[:-2]

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rates: torch.Tensor,
        normalize: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        rows: list[torch.Tensor] = []
        centre_rows: list[torch.Tensor] = []
        bandwidth_rows: list[torch.Tensor] = []
        for index in range(len(waveforms)):
            waveform = waveforms[index, : int(lengths[index])].flatten().float()
            rate = int(sample_rates[index])
            channels: list[torch.Tensor] = []
            centres = bandwidths = None
            for duration_ms in self.config.window_ms:
                image, centres, bandwidths = self._one_window(
                    waveform, rate, float(duration_ms)
                )
                channels.append(image)
            assert centres is not None and bandwidths is not None
            rows.append(torch.stack(channels, dim=1))  # [F,C,T]
            centre_rows.append(centres)
            bandwidth_rows.append(bandwidths)
        raw = torch.stack(rows, dim=0)
        mask = torch.ones(raw.shape[:2], dtype=torch.bool, device=raw.device)
        apply_normalization = self.config.normalize if normalize is None else bool(normalize)
        features = self.normalize_active(raw, mask) if apply_normalization else raw
        return {
            "features": features,
            "raw_features": raw,
            "valid_mask": mask,
            "centres_hz": torch.stack(centre_rows),
            "bandwidths_hz": torch.stack(bandwidth_rows),
        }


class PhysicalModulationFrontend(FeatureFrontend):
    """Native-rate physical-frequency envelopes and modulation channels.

    Analysis durations are specified in milliseconds; frequency centres and
    modulation rates are specified in hertz. Unobservable frequency rows are
    masked according to each recording's Nyquist frequency.
    """

    def __init__(self, config: PhysicalModulationConfig) -> None:
        super().__init__()
        self.config = config
        centres = torch.logspace(
            math.log10(max(config.minimum_hz, 1e-3)),
            math.log10(config.maximum_hz),
            config.frequency_bins,
        )
        sigma = max(1.0 / config.q_factor, 1e-4)
        half = math.sqrt(2.0 * math.log(2.0)) * sigma
        bandwidths = centres * (math.exp(half) - math.exp(-half))
        self.register_buffer("centres", centres, persistent=True)
        self.register_buffer("bandwidths", bandwidths, persistent=True)

    @property
    def output_channels(self) -> int:
        return len(self.config.window_ms) * (1 + len(self.config.modulation_rates_hz))

    def _stft_power(
        self, waveform: torch.Tensor, sample_rate: int, duration_ms: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        window_length = int(round(sample_rate * duration_ms / 1000.0))
        window_length = max(self.config.min_window_samples, window_length)
        window_length = min(self.config.max_window_samples, window_length)
        window_length = min(max(4, window_length), max(4, waveform.numel()))
        hop_length = max(1, int(round(0.25 * window_length)))
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
        return power, frequencies

    def _physical_projection(
        self,
        power: torch.Tensor,
        source_frequencies: torch.Tensor,
        target_frequencies: torch.Tensor,
    ) -> torch.Tensor:
        safe_source = source_frequencies.clamp_min(self.config.minimum_hz * 0.25)
        log_ratio = torch.log(
            safe_source[None] / target_frequencies[:, None].clamp_min(self.config.eps)
        )
        sigma = max(1.0 / self.config.q_factor, 1e-3)
        weights = torch.exp(-0.5 * (log_ratio / sigma).square())
        weights *= (source_frequencies[None] > 0.0).to(weights.dtype)
        weights /= weights.sum(dim=1, keepdim=True).clamp_min(self.config.eps)
        return weights @ power

    def _modulation_channels(
        self, envelope: torch.Tensor, duration_seconds: float
    ) -> list[torch.Tensor]:
        channels = [envelope]
        if not self.config.modulation_rates_hz:
            return channels
        frames = envelope.shape[-1]
        centred = envelope - envelope.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(centred, dim=-1)
        axis = torch.fft.rfftfreq(
            frames,
            d=duration_seconds / float(frames),
            device=envelope.device,
        ).to(envelope.dtype)
        safe_axis = axis.clamp_min(1.0 / max(duration_seconds, self.config.eps))
        for rate in self.config.modulation_rates_hz:
            response = torch.exp(-0.5 * (torch.log(safe_axis / rate) / 0.50).square())
            response *= (axis > 0).to(response.dtype)
            modulation = torch.fft.irfft(
                spectrum * response[None].to(spectrum.dtype), n=frames, dim=-1
            ).abs()
            channels.append(torch.log1p(modulation))
        return channels

    def _one(self, waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = waveform.flatten().float()
        if waveform.numel() < 8:
            waveform = F.pad(waveform, (0, 8 - waveform.numel()))
        duration_seconds = max(
            waveform.numel() / float(sample_rate), self.config.eps
        )
        target = self.centres.to(waveform)
        valid = target <= 0.5 * sample_rate * self.config.nyquist_rolloff
        if not valid.any():
            valid = valid.clone()
            valid[0] = True
        valid_target = target[valid]
        channels: list[torch.Tensor] = []
        for duration_ms in self.config.window_ms:
            power, source = self._stft_power(waveform, sample_rate, duration_ms)
            energy = self._physical_projection(power, source, valid_target)
            envelope = torch.log1p(torch.sqrt(energy.clamp_min(self.config.eps)))
            envelope = F.interpolate(
                envelope[None, None],
                size=(len(valid_target), self.config.time_bins),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            channels.extend(self._modulation_channels(envelope, duration_seconds))
        values = waveform.new_zeros(
            self.config.frequency_bins, self.output_channels, self.config.time_bins
        )
        values[valid] = torch.stack(channels, dim=1)
        return values, valid

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rates: torch.Tensor,
        normalize: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for index in range(len(waveforms)):
            values, mask = self._one(
                waveforms[index, : int(lengths[index])], int(sample_rates[index])
            )
            rows.append(values)
            masks.append(mask)
        raw = torch.stack(rows)
        valid_mask = torch.stack(masks).to(raw.device)
        apply_normalization = self.config.normalize if normalize is None else bool(normalize)
        features = self.normalize_active(raw, valid_mask) if apply_normalization else raw
        batch = raw.shape[0]
        return {
            "features": features,
            "raw_features": raw,
            "valid_mask": valid_mask,
            "centres_hz": self.centres.to(raw)[None].expand(batch, -1),
            "bandwidths_hz": self.bandwidths.to(raw)[None].expand(batch, -1),
        }
