// ABOUTME: Audio device clock rate estimator driven by PipeWire pw_time
// ABOUTME: Measures device-vs-system clock rate, which network sync cannot observe

use crate::log_sampling::should_log_sample;

/// Wall-clock span each short-window report covers.
const WINDOW_NS: i64 = 10_000_000_000;

/// Longest gap between samples before the span is considered contaminated.
/// Several quanta at any supported callback period (5-40 ms). Across such a
/// gap the device's ticks can stall or jump while wall time keeps advancing,
/// and the resulting error is too small for the plausibility bound below to
/// catch: a 500 ms tick stall inside a 10 s window reads -50,000 ppm, half
/// of MAX_PLAUSIBLE_PPM.
const MAX_SAMPLE_GAP_NS: i64 = 250_000_000;

/// Last-resort plausibility net, in the spirit of `TimeFilter::MAX_DRIFT`.
/// Deliberately loose: a bound tuned to a wired DAC could silently discard the
/// Bluetooth result this exists to find. Revisit once Bluetooth data exists.
const MAX_PLAUSIBLE_PPM: f64 = 100_000.0;

/// One `(ticks, now)` observation used as a measurement endpoint.
#[derive(Debug, Clone, Copy)]
struct Anchor {
    ticks: u64,
    now_ns: i64,
}

/// One emitted measurement.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DeviceClockReport {
    /// Device-vs-system clock rate over the last window, in ppm.
    pub short_ppm: f64,
    pub short_span_s: f64,
    /// Same, accumulated since the stream started or last reset.
    pub cumulative_ppm: f64,
    pub cumulative_span_s: f64,
    /// Discontinuities seen so far; see the reset rules.
    pub resets: u64,
}

/// Estimates the audio device's clock rate against the system clock.
///
/// `ClockSync` can only compare the server clock to the local system clock,
/// because that is all a network timestamp exchange can see. The rate that
/// actually drives playback error is the *device's* — a sound card crystal or,
/// over Bluetooth, a remote device across an A2DP transport. PipeWire reports
/// it: `ticks` is the device's sample counter and `now` the system-monotonic
/// time at which that count was valid, so their ratio is the missing term.
pub(crate) struct DeviceClockEstimator {
    short: Option<Anchor>,
    cumulative: Option<Anchor>,
    last: Option<Anchor>,
    rate: Option<(u32, u32)>,
    resets: u64,
}

/// Rate of `ticks` against wall time, as ppm away from nominal.
///
/// `rate` is a `spa_fraction` expressing the tick unit (normally 1/48000), so
/// nominal ticks per second is `denom / num`.
fn ppm_from(d_ticks: u64, d_now_ns: i64, rate_num: u32, rate_denom: u32) -> f64 {
    let measured = d_ticks as f64 * 1e9 / d_now_ns as f64;
    let nominal = f64::from(rate_denom) / f64::from(rate_num);
    (measured / nominal - 1.0) * 1e6
}

impl DeviceClockEstimator {
    pub fn new() -> Self {
        Self {
            short: None,
            cumulative: None,
            last: None,
            rate: None,
            resets: 0,
        }
    }

    /// Anchor all three endpoints on `sample` and record `rate`.
    /// Not a discontinuity by itself.
    fn anchor(&mut self, sample: Anchor, rate: (u32, u32)) {
        self.short = Some(sample);
        self.cumulative = Some(sample);
        self.last = Some(sample);
        self.rate = Some(rate);
    }

    /// Re-anchor every endpoint on `sample`. The span that was in flight
    /// described a timeline that no longer exists.
    fn reset(&mut self, sample: Anchor, rate: (u32, u32), cause: &str) {
        self.anchor(sample, rate);
        self.resets += 1;
        if should_log_sample(self.resets) {
            log::debug!("Device clock reset ({cause}); resets={}", self.resets);
        }
    }

    /// Feed one `pw_time` observation. Returns a report exactly when the short
    /// anchor's span reaches `WINDOW_NS`, re-anchoring at that point.
    pub fn update(
        &mut self,
        ticks: u64,
        now_ns: i64,
        rate_num: u32,
        rate_denom: u32,
    ) -> Option<DeviceClockReport> {
        if rate_num == 0 || rate_denom == 0 {
            return None;
        }
        let sample = Anchor { ticks, now_ns };
        let rate = (rate_num, rate_denom);

        // A format or quantum renegotiation makes prior ticks incomparable.
        if self.rate.is_some_and(|r| r != rate) {
            self.reset(sample, rate, "rate changed");
            return None;
        }

        let (Some(short), Some(cumulative), Some(last)) = (self.short, self.cumulative, self.last)
        else {
            self.anchor(sample, rate);
            return None;
        };

        let d_now = now_ns - last.now_ns;
        let d_ticks = i128::from(ticks) - i128::from(last.ticks);

        // A frozen pw_time report repeats ticks and now together: no new
        // information, but not a discontinuity either. A ticks-only freeze —
        // `now` advancing while the counter sits still — is skipped the same
        // way, and surfaces at resume as a gap against the retained `last`.
        // Keep the anchors and leave `last` alone so the next real sample
        // differences correctly.
        if d_now == 0 || d_ticks == 0 {
            return None;
        }
        if d_now < 0 || d_ticks < 0 {
            self.reset(sample, rate, "counter or clock went backwards");
            return None;
        }
        if d_now > MAX_SAMPLE_GAP_NS {
            self.reset(sample, rate, "stream was not scheduled");
            return None;
        }
        self.last = Some(sample);

        let short_span = now_ns - short.now_ns;
        if short_span < WINDOW_NS {
            return None;
        }

        let short_ppm = ppm_from(ticks - short.ticks, short_span, rate_num, rate_denom);
        if !short_ppm.is_finite() || short_ppm.abs() > MAX_PLAUSIBLE_PPM {
            self.reset(sample, rate, "implausible rate");
            return None;
        }

        let cumulative_span = now_ns - cumulative.now_ns;
        let report = DeviceClockReport {
            short_ppm,
            short_span_s: short_span as f64 / 1e9,
            cumulative_ppm: ppm_from(
                ticks - cumulative.ticks,
                cumulative_span,
                rate_num,
                rate_denom,
            ),
            cumulative_span_s: cumulative_span as f64 / 1e9,
            resets: self.resets,
        };
        self.short = Some(sample);
        Some(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOMINAL: f64 = 48_000.0;

    /// A synthetic device clock running `ppm` away from nominal 48 kHz.
    ///
    /// Ticks are derived from absolute elapsed time rather than accumulated
    /// per step, so rounding error stays bounded at ±1 tick over a window
    /// (≈2.1 ppm at a 10 s window) instead of growing with the step count.
    struct Fake {
        origin_ns: i64,
        origin_ticks: u64,
        now_ns: i64,
        ticks: u64,
        ppm: f64,
    }

    impl Fake {
        fn new(ppm: f64) -> Self {
            Self {
                origin_ns: 0,
                origin_ticks: 0,
                now_ns: 0,
                ticks: 0,
                ppm,
            }
        }

        /// Stream cycled: the device's counter restarts at the current time.
        fn restart_ticks(&mut self) {
            self.origin_ns = self.now_ns;
            self.origin_ticks = 0;
            self.ticks = 0;
        }

        /// Change the device rate from here on without a tick discontinuity.
        fn set_ppm(&mut self, ppm: f64) {
            self.origin_ns = self.now_ns;
            self.origin_ticks = self.ticks;
            self.ppm = ppm;
        }

        fn step(
            &mut self,
            est: &mut DeviceClockEstimator,
            cadence_ms: f64,
        ) -> Option<DeviceClockReport> {
            self.now_ns += (cadence_ms * 1e6) as i64;
            let elapsed_s = (self.now_ns - self.origin_ns) as f64 / 1e9;
            self.ticks =
                self.origin_ticks + (elapsed_s * NOMINAL * (1.0 + self.ppm / 1e6)).round() as u64;
            est.update(self.ticks, self.now_ns, 1, 48_000)
        }

        fn run(
            &mut self,
            est: &mut DeviceClockEstimator,
            secs: f64,
            cadence_ms: f64,
        ) -> Vec<DeviceClockReport> {
            let steps = (secs * 1000.0 / cadence_ms).round() as usize;
            (0..steps)
                .filter_map(|_| self.step(est, cadence_ms))
                .collect()
        }
    }

    /// The harness quantises ticks to integers, which is worth ±1 tick over a
    /// 10 s window ≈ 2.1 ppm. Assert to 3 ppm — still far tighter than the
    /// distinctions the measurement has to make (18-23 vs 2.5 vs 0).
    const PPM_TOLERANCE: f64 = 3.0;

    #[test]
    fn test_reports_the_device_rate_it_was_driven_at() {
        let mut est = DeviceClockEstimator::new();
        let mut dev = Fake::new(-18.0);
        let reports = dev.run(&mut est, 60.0, 21.3);
        assert!(!reports.is_empty(), "expected at least one report in 60s");
        let last = reports.last().unwrap();
        assert!(
            (last.short_ppm - -18.0).abs() < PPM_TOLERANCE,
            "short_ppm should recover -18ppm, got {:.2}",
            last.short_ppm
        );
        assert!(
            (last.cumulative_ppm - -18.0).abs() < PPM_TOLERANCE,
            "cumulative_ppm should recover -18ppm, got {:.2}",
            last.cumulative_ppm
        );
    }

    #[test]
    fn test_emits_one_report_per_window_not_per_sample() {
        let mut est = DeviceClockEstimator::new();
        let mut dev = Fake::new(0.0);
        let reports = dev.run(&mut est, 60.0, 21.3);
        assert!(
            (5..=6).contains(&reports.len()),
            "expected 5-6 reports in 60s at a 10s window, got {}",
            reports.len()
        );
    }

    #[test]
    fn test_no_report_before_a_full_window_has_elapsed() {
        let mut est = DeviceClockEstimator::new();
        let mut dev = Fake::new(-18.0);
        assert!(dev.run(&mut est, 9.0, 21.3).is_empty());
    }

    #[test]
    fn test_short_span_is_always_about_one_window() {
        let mut est = DeviceClockEstimator::new();
        let mut dev = Fake::new(-18.0);
        for r in dev.run(&mut est, 60.0, 21.3) {
            assert!(
                (r.short_span_s - 10.0).abs() < 0.1,
                "short_span_s should be ~10s, got {:.2}",
                r.short_span_s
            );
        }
    }

    #[test]
    fn test_repeated_report_is_skipped_not_treated_as_a_restart() {
        let mut est = DeviceClockEstimator::new();
        let mut dev = Fake::new(-18.0);
        dev.run(&mut est, 5.0, 21.3);
        // A frozen pw_time report repeats ticks AND now together.
        for _ in 0..50 {
            assert!(est.update(dev.ticks, dev.now_ns, 1, 48_000).is_none());
        }
        let reports = dev.run(&mut est, 8.0, 21.3);
        assert!(!reports.is_empty(), "a stall must not stop reporting");
        assert_eq!(reports[0].resets, 0, "a stall is not a reset");
        // On a fresh estimator both anchors are still the same sample, so the
        // two spans are bit-identical: same `Anchor`, same subtraction. Any
        // re-anchoring of `cumulative` during the stall breaks the equality
        // wherever the stall fell, which a span threshold would not.
        assert_eq!(
            reports[0].cumulative_span_s, reports[0].short_span_s,
            "on a fresh estimator the first report's anchors are the same sample"
        );
        assert!(
            (reports[0].short_ppm - -18.0).abs() < PPM_TOLERANCE,
            "the retained anchor must still measure the real rate, got {:.2}",
            reports[0].short_ppm
        );
    }

    #[test]
    fn test_a_long_tick_freeze_resets_on_resume() {
        // Ticks frozen while `now` advances: each sample is skipped, so `last`
        // stays at the last distinct sample and the resume trips the gap rule.
        let mut est = DeviceClockEstimator::new();
        let mut dev = Fake::new(-18.0);
        dev.run(&mut est, 5.0, 21.3);
        let frozen = dev.ticks;
        for _ in 0..20 {
            dev.now_ns += 21_300_000;
            assert!(est.update(frozen, dev.now_ns, 1, 48_000).is_none());
        }
        assert_eq!(est.resets, 0, "a freeze in progress is not yet a reset");
        assert!(dev.step(&mut est, 21.3).is_none());
        assert_eq!(est.resets, 1, "426ms of frozen ticks must reset on resume");
    }

    #[test]
    fn test_ticks_restart_resets_and_emits_no_garbage() {
        let mut est = DeviceClockEstimator::new();
        let mut dev = Fake::new(-18.0);
        dev.run(&mut est, 30.0, 21.3);
        dev.restart_ticks();
        assert!(
            dev.step(&mut est, 21.3).is_none(),
            "restart should reset, not report"
        );
        let reports = dev.run(&mut est, 12.0, 21.3);
        assert!(
            !reports.is_empty(),
            "should resume reporting after re-anchoring"
        );
        assert_eq!(reports[0].resets, 1);
        assert!(
            reports[0].short_ppm.abs() < 1000.0,
            "post-restart rate must not be garbage, got {:.0}ppm",
            reports[0].short_ppm
        );
    }

    #[test]
    fn test_scheduling_gap_resets() {
        let mut est = DeviceClockEstimator::new();
        let mut dev = Fake::new(-18.0);
        dev.run(&mut est, 5.0, 21.3);
        // Not scheduled for 500ms. `Fake` advances ticks proportionally, so
        // this is the benign variant — the rate across the gap would have been
        // right. The reset fires on elapsed time regardless of what ticks did,
        // which is the point: the rule cannot know which variant it got.
        assert!(dev.step(&mut est, 500.0).is_none());
        let reports = dev.run(&mut est, 12.0, 21.3);
        assert_eq!(
            reports
                .first()
                .expect("should report after re-anchoring")
                .resets,
            1
        );
    }

    #[test]
    fn test_rate_change_resets() {
        let mut est = DeviceClockEstimator::new();
        let mut dev = Fake::new(-18.0);
        dev.run(&mut est, 30.0, 21.3);
        dev.now_ns += 21_300_000;
        assert!(
            est.update(dev.ticks + 44, dev.now_ns, 1, 44_100).is_none(),
            "a rate change must reset, not report"
        );
        assert_eq!(est.resets, 1);
    }

    #[test]
    fn test_implausible_rate_is_rejected() {
        let mut est = DeviceClockEstimator::new();
        let mut dev = Fake::new(200_000.0); // 20%, twice the plausibility bound
        assert!(
            dev.run(&mut est, 40.0, 21.3).is_empty(),
            "an implausible rate must never be reported"
        );
        assert!(est.resets > 0, "it should have reset instead");
    }

    #[test]
    fn test_backwards_time_resets() {
        let mut est = DeviceClockEstimator::new();
        let mut dev = Fake::new(-18.0);
        dev.run(&mut est, 5.0, 21.3);
        assert!(est
            .update(dev.ticks + 1024, dev.now_ns - 1_000_000, 1, 48_000)
            .is_none());
        assert_eq!(est.resets, 1);
    }

    #[test]
    fn test_short_window_tracks_a_rate_change_before_cumulative_does() {
        let mut est = DeviceClockEstimator::new();
        let mut dev = Fake::new(-18.0);
        dev.run(&mut est, 40.0, 21.3);
        dev.set_ppm(-100.0);
        let reports = dev.run(&mut est, 25.0, 21.3);
        let last = reports
            .last()
            .expect("expected reports after the rate change");
        assert!(
            (last.short_ppm - -100.0).abs() < PPM_TOLERANCE,
            "short window should follow the new rate, got {:.2}",
            last.short_ppm
        );
        assert!(
            last.cumulative_ppm > -100.0 && last.cumulative_ppm < -18.0,
            "cumulative should sit between the two rates, got {:.2}",
            last.cumulative_ppm
        );
    }

    #[test]
    fn test_zero_rate_fraction_is_ignored() {
        // A zero numerator or denominator would divide by zero in ppm_from.
        let mut est = DeviceClockEstimator::new();
        assert!(est.update(1_000, 1_000_000, 0, 48_000).is_none());
        assert!(
            est.short.is_none() && est.rate.is_none(),
            "a zero rate must not arm the estimator"
        );
        assert!(est.update(1_000, 1_000_000, 1, 0).is_none());
        assert!(
            est.short.is_none() && est.rate.is_none(),
            "a zero denominator must not arm the estimator either"
        );
        assert_eq!(est.resets, 0, "an ignored sample is not a discontinuity");
    }

    #[test]
    fn test_non_unit_rate_fraction_is_handled() {
        // `rate` is a fraction, not a bare denominator: 2/96000 is the same
        // 48kHz tick rate as 1/48000, so a device at nominal reads ~0ppm.
        let mut est = DeviceClockEstimator::new();
        let mut now_ns: i64 = 0;
        let mut out = None;
        for i in 0..600 {
            let ticks = (i as f64 * 0.0213 * 48_000.0).round() as u64;
            if let Some(r) = est.update(ticks, now_ns, 2, 96_000) {
                out = Some(r);
            }
            now_ns += 21_300_000;
        }
        let r = out.expect("expected a report at a non-unit rate fraction");
        assert!(
            r.short_ppm.abs() < PPM_TOLERANCE,
            "2/96000 should read as nominal, got {:.2}ppm",
            r.short_ppm
        );
    }
}
