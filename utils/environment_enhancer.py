"""
Environment Enhancer for VIGIN_ai.
Provides real-time pre-filtering for low-light, rain, fog, and adverse weather conditions:
- Auto condition detection via luminance, contrast, and dark channel haze metrics.
- Adaptive Retinex/gamma boost + CLAHE on Y channel with headlight highlight protection for night scenes.
- Optimized Dark Channel Prior (DCP) de-hazing + bilateral filtering for rain streaks and atmospheric noise.
"""

import cv2
import numpy as np
from collections import deque


class EnvironmentEnhancer:
    """
    Real-time video pre-filtering engine for adverse weather and illumination.
    Modes:
        'auto'  : Automatically senses environmental conditions and applies appropriate filters.
        'night' : Forces low-light Retinex/gamma boost + CLAHE with glare preservation.
        'rain'  : Forces Dark Channel Prior (DCP) de-hazing + bilateral rain streak suppression.
        'clear' : Direct pass-through or subtle micro-contrast enhancement.
    """

    def __init__(
        self,
        mode="auto",
        luminance_threshold=75.0,
        contrast_threshold=38.0,
        haze_threshold=0.28,
        auto_interval=15
    ):
        self.mode = mode.lower() if mode else "auto"
        self.luminance_threshold = float(luminance_threshold)
        self.contrast_threshold = float(contrast_threshold)
        self.haze_threshold = float(haze_threshold)
        self.auto_interval = int(auto_interval)

        # State tracking
        self.active_condition = "clear" if self.mode != "auto" else "clear"
        self.frame_counter = 0
        self.history_len = 5
        self.lum_history = deque(maxlen=self.history_len)
        self.contrast_history = deque(maxlen=self.history_len)
        self.haze_history = deque(maxlen=self.history_len)

        # Cached diagnostic metrics
        self.latest_metrics = {
            "luminance": 0.0,
            "contrast": 0.0,
            "haze": 0.0,
            "condition": self.active_condition,
            "applied_mode": self.mode
        }

        # Pre-configured CLAHE objects for efficiency
        self.clahe_night = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        self.clahe_subtle = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))

    def reset_history(self):
        """Clears temporal smoothing buffers."""
        self.lum_history.clear()
        self.contrast_history.clear()
        self.haze_history.clear()
        self.frame_counter = 0

    def analyze_environment(self, frame):
        """
        Analyzes scene metrics (luminance, standard deviation contrast, and dark channel haze).
        Uses a downscaled proxy frame to minimize computational latency.
        """
        fh, fw = frame.shape[:2]
        # Downsample to 240p for ultra-fast metric extraction (~1ms)
        proxy_w = 320
        proxy_h = max(180, int(fh * (proxy_w / fw)))
        small = cv2.resize(frame, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)

        # 1. Luminance & Contrast via Y channel (YCrCb)
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        mean_lum = float(np.mean(y_channel))
        contrast_std = float(np.std(y_channel))

        # 2. Dark Channel Haze Metric
        # In foggy/rainy atmospheres, the dark channel intensity rises significantly
        min_channel = np.min(small, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dark_channel = cv2.erode(min_channel, kernel)
        mean_dark = float(np.mean(dark_channel)) / 255.0

        # Push to rolling histories for temporal hysteresis
        self.lum_history.append(mean_lum)
        self.contrast_history.append(contrast_std)
        self.haze_history.append(mean_dark)

        smoothed_lum = float(np.mean(self.lum_history))
        smoothed_contrast = float(np.mean(self.contrast_history))
        smoothed_haze = float(np.mean(self.haze_history))

        # Condition Classification Logic:
        # - Night/Low-Light: Scene overall is dark (luminance < luminance_threshold)
        # - Rain/Haze/Fog: Dark channel is elevated, contrast is diminished, but scene is not pure night
        # - Clear: Balanced illumination and crisp contrast
        if smoothed_lum < self.luminance_threshold:
            detected = "night"
        elif smoothed_haze > self.haze_threshold and smoothed_contrast < 58.0:
            detected = "rain"
        else:
            detected = "clear"

        self.latest_metrics = {
            "luminance": round(smoothed_lum, 1),
            "contrast": round(smoothed_contrast, 1),
            "haze": round(smoothed_haze, 3),
            "condition": detected,
            "applied_mode": self.mode
        }

        return detected, self.latest_metrics

    def enhance_night(self, frame):
        """
        Adaptive Retinex / Gamma Boost with CLAHE and Highlight Preservation.
        Lifts deep shadows and dark pavement while protecting headlights/streetlights from blowing out.
        """
        if frame is None or frame.size == 0:
            return frame

        # Convert to YCrCb to boost brightness and local contrast without shifting chroma
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        mean_y = float(np.mean(y))
        # Target balanced luminance ~120. Compute adaptive gamma
        # For very dark scenes (mean_y=30), gamma < 1.0 brightens values.
        norm_mean = max(15.0, min(mean_y, 220.0)) / 255.0
        target_norm = 125.0 / 255.0
        adaptive_gamma = float(np.clip(np.log(target_norm) / np.log(norm_mean), 0.45, 0.90))

        # Build fast lookup table for gamma curve
        gamma_lut = np.array([
            int(np.clip(((i / 255.0) ** adaptive_gamma) * 255.0, 0, 255))
            for i in range(256)
        ], dtype=np.uint8)

        boosted_y = cv2.LUT(y, gamma_lut)

        # Apply CLAHE to restore local micro-textures on road markings & vehicle contours
        clahe_y = self.clahe_night.apply(boosted_y)

        # Highlight Protection Mask:
        # Headlights and taillights are already high luminance (> 210).
        # We blend the original Y into bright regions to prevent headlight bloom/washout.
        highlight_thresh = 210
        if np.max(y) > highlight_thresh:
            mask = np.clip((y.astype(np.float32) - highlight_thresh) / (255.0 - highlight_thresh), 0.0, 1.0)
            final_y = (clahe_y.astype(np.float32) * (1.0 - mask) + y.astype(np.float32) * mask).astype(np.uint8)
        else:
            final_y = clahe_y

        merged = cv2.merge((final_y, cr, cb))
        enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        return enhanced_bgr

    def enhance_rain(self, frame):
        """
        High-Speed Dark Channel Prior (DCP) De-hazing with Bilateral Rain-Streak Suppression.
        Operates in YCrCb color space for sub-25ms real-time throughput.
        """
        if frame is None or frame.size == 0:
            return frame

        fh, fw = frame.shape[:2]

        # Convert to YCrCb: de-hazing acts on luminance (Y) while preserving color fidelity (Cr, Cb)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # 1. Downscale Y channel to compute dark channel and atmospheric light quickly
        scale_w = 320
        scale_h = max(180, int(fh * (scale_w / fw)))
        small_y = cv2.resize(y, (scale_w, scale_h), interpolation=cv2.INTER_AREA)

        # Dark channel of luminance
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dark = cv2.erode(small_y, kernel)

        # 2. Atmospheric light estimation (A)
        flat_dark = dark.reshape(-1)
        num_top = max(10, int(len(flat_dark) * 0.001))
        top_indices = np.argpartition(flat_dark, -num_top)[-num_top:]
        a_lum = float(np.max(small_y.reshape(-1)[top_indices]))
        a_lum = float(np.clip(a_lum, 160.0, 245.0))

        # 3. Fast Transmission Map estimation: t = 1 - omega * (Y / A)
        omega = 0.80
        norm_y = small_y.astype(np.float32) / (a_lum + 1e-5)
        trans_small = np.clip(1.0 - omega * cv2.erode(norm_y, kernel), 0.20, 1.0)
        trans_full = cv2.resize(trans_small, (fw, fh), interpolation=cv2.INTER_LINEAR)

        # 4. Scene Radiance Recovery on Y channel
        y_f = y.astype(np.float32)
        recovered_y = np.clip((y_f - a_lum) / trans_full + a_lum, 0.0, 255.0).astype(np.uint8)

        # 5. Bilateral Filter on Luminance for Rain Streak and Splash Denoising
        denoised_y = cv2.bilateralFilter(recovered_y, d=5, sigmaColor=28, sigmaSpace=28)

        # 6. Contrast restoration via CLAHE
        res_y = self.clahe_subtle.apply(denoised_y)

        # Recombine with Cr, Cb and return to BGR
        result = cv2.cvtColor(cv2.merge((res_y, cr, cb)), cv2.COLOR_YCrCb2BGR)
        return result

    def process(self, frame):
        """
        Processes an incoming video frame through the environment enhancement pipeline.

        Returns:
            enhanced_frame (np.ndarray): Enhanced BGR image ready for detection heads.
            condition (str): Active condition ('night', 'rain', 'clear').
        """
        if frame is None or frame.size == 0:
            return frame, self.active_condition

        self.frame_counter += 1

        # Periodic condition sensing in 'auto' mode
        if self.mode == "auto":
            if self.frame_counter % self.auto_interval == 1 or not self.lum_history:
                detected, _ = self.analyze_environment(frame)
                self.active_condition = detected
        else:
            self.active_condition = self.mode

        # Dispatch enhancement filter
        if self.active_condition == "night":
            enhanced = self.enhance_night(frame)
        elif self.active_condition == "rain":
            enhanced = self.enhance_rain(frame)
        else:
            enhanced = frame

        return enhanced, self.active_condition

    def get_condition_info(self):
        """Returns the current condition and diagnostic metrics."""
        return {
            "active_condition": self.active_condition,
            "configured_mode": self.mode,
            "metrics": self.latest_metrics
        }
