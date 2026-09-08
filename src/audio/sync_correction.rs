// ABOUTME: Sync correction planner for drop/insert cadence, plus the
// ABOUTME: measurement filter and engage gate that keep it off phantom errors

/// Correction schedule for drop/insert cadence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CorrectionSchedule {
    /// Insert one frame every N frames (0 disables).
    pub insert_every_n_frames: u32,
    /// Drop one frame every N frames (0 disables).
    pub drop_every_n_frames: u32,
    /// True when re-anchoring is required.
    pub reanchor: bool,
}

impl CorrectionSchedule {
    /// True when any correction (insert, drop, or reanchor) is active.
    pub fn is_correcting(&self) -> bool {
        self.insert_every_n_frames > 0 || self.drop_every_n_frames > 0 || self.reanchor
    }
}

/// Planner that converts sync error into a correction schedule.
///
/// Uses hysteresis to prevent oscillation at the deadband boundary:
/// correction engages at `engage_us` and disengages at `deadband_us`.
#[derive(Debug, Clone, Copy)]
pub struct CorrectionPlanner {
    deadband_us: i64,
    engage_us: i64,
    reanchor_threshold_us: i64,
    target_seconds: f64,
    max_speed_correction: f64,
}

impl CorrectionPlanner {
    /// Create a planner with default thresholds.
    pub fn new() -> Self {
        Self {
            deadband_us: 1_500,
            engage_us: 3_000,
            reanchor_threshold_us: 500_000,
            target_seconds: 2.0,
            max_speed_correction: 0.04,
        }
    }

    /// Plan a correction schedule from sync error and sample rate.
    ///
    /// `error_us` is the sync error in microseconds (positive = ahead, negative = behind).
    /// `currently_correcting` controls hysteresis: when already correcting, the lower
    /// deadband threshold is used instead of the engage threshold.
    pub fn plan(
        &self,
        error_us: i64,
        sample_rate: u32,
        currently_correcting: bool,
    ) -> CorrectionSchedule {
        let abs_error = error_us.saturating_abs();

        // Hysteresis: use lower threshold to keep correcting, higher to start.
        let threshold = if currently_correcting {
            self.deadband_us
        } else {
            self.engage_us
        };

        if abs_error <= threshold {
            return CorrectionSchedule {
                insert_every_n_frames: 0,
                drop_every_n_frames: 0,
                reanchor: false,
            };
        }

        if abs_error >= self.reanchor_threshold_us {
            return CorrectionSchedule {
                insert_every_n_frames: 0,
                drop_every_n_frames: 0,
                reanchor: true,
            };
        }

        let sample_rate_f = sample_rate as f64;
        let frames_error = (error_us as f64 * sample_rate_f) / 1_000_000.0;
        let desired_corrections_per_sec = frames_error.abs() / self.target_seconds;
        let max_corrections_per_sec = sample_rate_f * self.max_speed_correction;
        let corrections_per_sec = desired_corrections_per_sec.min(max_corrections_per_sec);

        if corrections_per_sec <= 0.0 {
            return CorrectionSchedule {
                insert_every_n_frames: 0,
                drop_every_n_frames: 0,
                reanchor: false,
            };
        }

        let interval_frames = (sample_rate_f / corrections_per_sec).round() as u32;

        if error_us > 0 {
            CorrectionSchedule {
                insert_every_n_frames: 0,
                drop_every_n_frames: interval_frames.max(1),
                reanchor: false,
            }
        } else {
            CorrectionSchedule {
                insert_every_n_frames: interval_frames.max(1),
                drop_every_n_frames: 0,
                reanchor: false,
            }
        }
    }
}

impl Default for CorrectionPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Upper bound on measurements kept by [`SyncErrorFilter`]. 101 callbacks is
/// ~1s at the common 10ms WASAPI period (2-4s on 20-40ms periods): long enough
/// that the floor mode gets sampled even through dense flapping, short enough
/// that a real displacement reaches the output within about a second.
///
/// This is a *sample* count, so on backends with a long callback period it no
/// longer describes the same amount of time — a PipeWire stream that is handed
/// a 12288-frame quantum wakes every 256ms, which would stretch the window to
/// ~26s. See [`SYNC_ERROR_WINDOW_SPAN_US`].
pub(crate) const SYNC_ERROR_WINDOW: usize = 101;

/// Wall-clock span the floor window is allowed to cover.
///
/// The window exists to outlast padding-jitter flapping, which is a
/// *time*-domain phenomenon; sizing it purely in callbacks let a long callback
/// period stretch it far past what the loop can stay stable with. The floor
/// lags a rising error by one whole window, so that lag is dead time in the
/// correction loop: once it grows past a few seconds the loop overshoots every
/// episode and limit-cycles, audibly (measured on PipeWire at a 256ms
/// callback: a 27s 0.18% episode injecting 46ms of opposite error, cleared by
/// a 7s 1.27% episode, repeating every ~148s).
///
/// 4s is the top of the span the sample cap already produced on supported
/// backends ("2-4s on 20-40ms periods"), so this bound changes nothing for
/// them — 10ms and 20ms callbacks still clamp to 101 samples and 40ms lands on
/// 100 — while holding every longer period to the same 4s. Swept against the
/// closed loop from 5ms to 512ms it keeps the worst applied correction at
/// 0.21%, roughly a thirtieth of a semitone; the unbounded sample count
/// reaches 1.27% at 256ms and 2.78% at 512ms.
pub(crate) const SYNC_ERROR_WINDOW_SPAN_US: u64 = 4_000_000;

/// Floor over fewer than this many samples is too easy for a single outlier to
/// dominate, so the span bound never shrinks the window below it.
pub(crate) const MIN_SYNC_ERROR_WINDOW: usize = 8;

/// Number of samples whose span stays within [`SYNC_ERROR_WINDOW_SPAN_US`] at
/// the given callback period, clamped to the supported window range.
pub(crate) fn window_len_for_period(period_us: u64) -> usize {
    if period_us == 0 {
        return SYNC_ERROR_WINDOW;
    }
    let fits = SYNC_ERROR_WINDOW_SPAN_US.div_ceil(period_us) as usize;
    fits.clamp(MIN_SYNC_ERROR_WINDOW, SYNC_ERROR_WINDOW)
}

/// Windowed minimum (floor tracker) over recent sync-error measurements.
///
/// Audio presentation timestamps are quantized to the engine period: under
/// scheduler jitter the wake-time padding snapshot jumps *up* by whole
/// periods (observed as a 2ms <-> 12ms alternation on shared-mode WASAPI)
/// while the endpoint FIFO plays gaplessly. The noise is one-sided — extra
/// queued padding can only make a reading later, never earlier — so the
/// window floor is the honest alignment estimate: it holds regardless of
/// which mode has the majority, so long as the floor mode is sampled at
/// least once per window. A real displacement (engine starvation inserting
/// silence, a cursor jump) shifts every reading *including the floor*, so it
/// still reaches the planner within one window.
///
/// The response is deliberately asymmetric: a drop in error propagates
/// immediately (a new low sample becomes the floor at once, so a running
/// correction converges without overshoot), while a rise propagates only
/// once the old floor ages out of the window — exactly the conservatism an
/// engage decision wants. A spuriously *low* outlier would bias toward
/// under-correction, the safe direction. The padding term never produces
/// one (padding cannot go below empty); the clock filter normally moves by
/// microseconds once settled, though an outlier time sample can still step
/// the estimate by low single-digit milliseconds in either direction — a
/// negative step costs at most one window of under-correction.
///
/// Allocation-free, O(window) scan per update; sized for the audio callback.
#[derive(Debug, Clone)]
pub(crate) struct SyncErrorFilter {
    samples: [i64; SYNC_ERROR_WINDOW],
    len: usize,
    next: usize,
    /// Live window length: `SYNC_ERROR_WINDOW` until a callback period is
    /// observed, then whatever fits in [`SYNC_ERROR_WINDOW_SPAN_US`].
    window: usize,
}

impl SyncErrorFilter {
    /// Create an empty (cold) filter.
    pub fn new() -> Self {
        Self {
            samples: [0; SYNC_ERROR_WINDOW],
            len: 0,
            next: 0,
            window: SYNC_ERROR_WINDOW,
        }
    }

    /// Size the window to the observed callback period.
    ///
    /// Cheap and idempotent, so the audio callback can call it every wake; a
    /// period change discards the history, since samples taken at a different
    /// wake cadence describe a different window.
    pub fn set_callback_period(&mut self, period_us: u64) {
        let window = window_len_for_period(period_us);
        if window != self.window {
            self.window = window;
            self.reset();
        }
    }

    /// Discard all samples, e.g. after a reanchor or generation change made
    /// prior measurements meaningless.
    pub fn reset(&mut self) {
        self.len = 0;
        self.next = 0;
    }

    /// True once the window is fully populated. A floor over a partial
    /// window is still returned by [`SyncErrorFilter::update`], but engage
    /// decisions should wait for warmth (see [`EngageGate`]).
    pub fn is_warm(&self) -> bool {
        self.len == self.window
    }

    /// Record a raw error measurement (µs) and return the windowed minimum.
    pub fn update(&mut self, raw_error_us: i64) -> i64 {
        self.samples[self.next] = raw_error_us;
        self.next = (self.next + 1) % self.window;
        if self.len < self.window {
            self.len += 1;
        }
        // While filling, the live samples are the contiguous prefix
        // (next == len until the first wrap); afterwards the whole array.
        // Stale slots past `len` are never read. `len >= 1` here, so the
        // fold always sees at least one real sample.
        self.samples[..self.len]
            .iter()
            .copied()
            .fold(i64::MAX, i64::min)
    }
}

impl Default for SyncErrorFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Consecutive correcting plans required before a correction may engage from
/// idle: ~0.5s at a 10ms callback period. Filtered errors already move at
/// window timescale, so a genuine displacement holds the streak trivially;
/// the gate keeps engagement evidence-backed across filter resets and
/// warm-up, when the estimate is least trustworthy.
pub(crate) const ENGAGE_STREAK_PLANS: u32 = 50;

/// Gate between the planner and the applied schedule: an idle stream only
/// starts correcting after a sustained run of correcting plans over a warm
/// filter. Every engaged correction mutates real audio (dropped or repeated
/// frames are audible as ticks), so engagement must be evidence-backed;
/// disengagement and replanning of a running correction stay immediate
/// because stopping or retuning is never audible.
#[derive(Debug, Clone)]
pub(crate) struct EngageGate {
    streak: u32,
}

impl EngageGate {
    /// Create a gate with no accumulated streak.
    pub fn new() -> Self {
        Self { streak: 0 }
    }

    /// Clear the accumulated streak.
    pub fn reset(&mut self) {
        self.streak = 0;
    }

    /// Admit or suppress a planned schedule.
    ///
    /// `currently_correcting` is whether a correction episode is already
    /// running; `filter_warm` is whether the error filter window is fully
    /// populated. Reanchor plans always pass: they indicate gross real
    /// displacement where a delayed reaction is worse than a rare false
    /// positive.
    pub fn admit(
        &mut self,
        planned: CorrectionSchedule,
        currently_correcting: bool,
        filter_warm: bool,
    ) -> CorrectionSchedule {
        if planned.reanchor || currently_correcting || !planned.is_correcting() {
            self.streak = 0;
            return planned;
        }
        self.streak = self.streak.saturating_add(1);
        if filter_warm && self.streak >= ENGAGE_STREAK_PLANS {
            planned
        } else {
            CorrectionSchedule::default()
        }
    }
}

impl Default for EngageGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_correction_within_engage_threshold() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(2_500, 48_000, false);
        assert!(!schedule.is_correcting(), "should not engage below 3ms");
    }

    #[test]
    fn test_correction_engages_above_threshold() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(3_500, 48_000, false);
        assert!(schedule.is_correcting(), "should engage above 3ms");
        assert!(schedule.drop_every_n_frames > 0, "positive error = drop");
    }

    #[test]
    fn test_hysteresis_keeps_correcting_above_deadband() {
        let planner = CorrectionPlanner::new();
        // 2ms error: below engage (3ms) but above deadband (1.5ms).
        // Should keep correcting if already active.
        let schedule = planner.plan(2_000, 48_000, true);
        assert!(
            schedule.is_correcting(),
            "should keep correcting above 1.5ms deadband"
        );
    }

    #[test]
    fn test_hysteresis_stops_below_deadband() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(1_000, 48_000, true);
        assert!(
            !schedule.is_correcting(),
            "should stop below 1.5ms deadband"
        );
    }

    #[test]
    fn test_negative_error_inserts() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(-5_000, 48_000, false);
        assert!(
            schedule.insert_every_n_frames > 0,
            "negative error = insert"
        );
        assert_eq!(schedule.drop_every_n_frames, 0);
    }

    #[test]
    fn test_reanchor_at_large_error() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(500_000, 48_000, false);
        assert!(schedule.reanchor);
    }

    #[test]
    fn test_exact_engage_threshold_does_not_engage() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(3_000, 48_000, false);
        assert!(
            !schedule.is_correcting(),
            "exactly at engage threshold should not engage (<=)"
        );
    }

    #[test]
    fn test_exact_deadband_threshold_disengages() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(1_500, 48_000, true);
        assert!(
            !schedule.is_correcting(),
            "exactly at deadband threshold should disengage (<=)"
        );
    }

    #[test]
    fn test_negative_hysteresis_keeps_inserting_above_deadband() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(-2_000, 48_000, true);
        assert!(
            schedule.is_correcting(),
            "should keep correcting negative error above deadband"
        );
        assert!(
            schedule.insert_every_n_frames > 0,
            "negative error = insert"
        );
        assert_eq!(schedule.drop_every_n_frames, 0);
    }

    // -- Exact interval values --
    //
    // The tests above only check *which* field is set (drop vs insert).
    // These pin down the actual interval arithmetic: `frames_error`, the
    // `desired` / `max` corrections_per_sec cap, and the `interval_frames`
    // rounding. Checking a single "correction engaged" bit lets the
    // arithmetic drift silently; checking the computed cadence doesn't.

    #[test]
    fn test_drop_interval_below_speed_cap_matches_spec() {
        // 5ms error at 48 kHz, already correcting (deadband threshold):
        //   frames_error                = 5000 * 48000 / 1_000_000 = 240 frames
        //   desired_corrections_per_sec = 240 / 2.0                = 120
        //   max_corrections_per_sec     = 48000 * 0.04             = 1920 (not binding)
        //   interval_frames             = round(48000 / 120)       = 400
        let planner = CorrectionPlanner::new();
        let s = planner.plan(5_000, 48_000, true);
        assert_eq!(
            s.drop_every_n_frames, 400,
            "5ms drift should produce drop every 400 frames"
        );
        assert_eq!(s.insert_every_n_frames, 0);
    }

    #[test]
    fn test_insert_interval_below_speed_cap_matches_spec() {
        // Mirror of the drop test with a negative error.
        let planner = CorrectionPlanner::new();
        let s = planner.plan(-5_000, 48_000, true);
        assert_eq!(
            s.insert_every_n_frames, 400,
            "-5ms drift should produce insert every 400 frames"
        );
        assert_eq!(s.drop_every_n_frames, 0);
    }

    #[test]
    fn test_drop_interval_hits_max_speed_cap() {
        // 200ms error — large enough that the desired rate exceeds the cap:
        //   frames_error                = 200000 * 48000 / 1_000_000 = 9600
        //   desired_corrections_per_sec = 9600 / 2.0                 = 4800
        //   max_corrections_per_sec     = 48000 * 0.04               = 1920 (binding)
        //   interval_frames             = round(48000 / 1920)        = 25
        let planner = CorrectionPlanner::new();
        let s = planner.plan(200_000, 48_000, true);
        assert_eq!(
            s.drop_every_n_frames, 25,
            "large drift should be capped at max speed correction (interval = 25 frames)"
        );
    }

    #[test]
    fn test_filter_constant_input_passes_through() {
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW {
            assert_eq!(filter.update(4_000), 4_000);
        }
        assert!(filter.is_warm());
    }

    #[test]
    fn test_filter_warms_only_after_full_window() {
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW - 1 {
            filter.update(0);
            assert!(!filter.is_warm());
        }
        filter.update(0);
        assert!(filter.is_warm());
    }

    #[test]
    fn test_filter_floor_ignores_flap_at_any_duty() {
        // Period-quantized flapping is one-sided (+one period). Even a 90%
        // majority in the high mode — enough to drag any central-tendency
        // estimate — must not lift the floor while the quiet mode is sampled
        // at least once per window.
        let mut filter = SyncErrorFilter::new();
        for i in 0..SYNC_ERROR_WINDOW * 3 {
            let raw = if i % 10 == 0 { 0 } else { 10_000 };
            let floor = filter.update(raw);
            if i >= 1 {
                assert_eq!(floor, 0, "flap majority must not lift the floor");
            }
        }
    }

    #[test]
    fn test_filter_floor_follows_persistent_rise_after_full_window() {
        // A real displacement shifts every reading including the floor; the
        // output must follow once the old floor ages out of the window.
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW {
            filter.update(0);
        }
        let mut floor = 0;
        for i in 0..SYNC_ERROR_WINDOW {
            floor = filter.update(10_000);
            if i < SYNC_ERROR_WINDOW - 1 {
                assert_eq!(floor, 0, "rise must wait for the window to age out");
            }
        }
        assert_eq!(floor, 10_000, "persistent rise must reach the output");
    }

    #[test]
    fn test_filter_floor_follows_drop_immediately() {
        // A running correction shrinks the error; the floor must track the
        // new low on the very next sample so the correction converges
        // without overshoot.
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW {
            filter.update(10_000);
        }
        assert_eq!(filter.update(2_000), 2_000, "new low becomes the floor");
    }

    #[test]
    fn test_filter_handles_negative_errors() {
        // Negative true error (audio early, insert side) with one-sided
        // positive excursions on top.
        let mut filter = SyncErrorFilter::new();
        for i in 0..SYNC_ERROR_WINDOW {
            let raw = if i % 3 == 0 { 10_000 } else { -5_000 };
            filter.update(raw);
        }
        assert_eq!(filter.update(-5_000), -5_000);
    }

    #[test]
    fn test_filter_single_low_evicted_exactly_after_window() {
        // One low among highs must stop being the floor exactly WINDOW
        // updates after it was recorded.
        let mut filter = SyncErrorFilter::new();
        filter.update(1_000);
        for i in 0..SYNC_ERROR_WINDOW - 1 {
            assert_eq!(filter.update(10_000), 1_000, "low still in window at {i}");
        }
        assert_eq!(
            filter.update(10_000),
            10_000,
            "low must age out exactly one window after it was recorded"
        );
    }

    #[test]
    fn test_filter_tracks_min_across_multiple_wraps() {
        // Distinct values across several ring cycles: the floor must always
        // equal the true minimum of the last WINDOW samples.
        let mut filter = SyncErrorFilter::new();
        let mut recent: Vec<i64> = Vec::new();
        for i in 0..(SYNC_ERROR_WINDOW as i64 * 5) {
            let value = (i * 37) % 1_000 + if i % 13 == 0 { -500 } else { 0 };
            recent.push(value);
            if recent.len() > SYNC_ERROR_WINDOW {
                recent.remove(0);
            }
            let expected = *recent.iter().min().unwrap();
            assert_eq!(filter.update(value), expected, "mismatch at update {i}");
        }
    }

    #[test]
    fn test_filter_reset_clears_window() {
        // Fill with a LOW value so stale slots would drag the floor down if
        // reset failed to fence them off.
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW {
            filter.update(-10_000);
        }
        filter.reset();
        assert!(!filter.is_warm());
        assert_eq!(
            filter.update(5_000),
            5_000,
            "floor after reset must reflect only new samples"
        );
    }

    fn correcting_plan() -> CorrectionSchedule {
        CorrectionSchedule {
            insert_every_n_frames: 0,
            drop_every_n_frames: 200,
            reanchor: false,
        }
    }

    #[test]
    fn test_gate_requires_sustained_streak() {
        let mut gate = EngageGate::new();
        for _ in 0..ENGAGE_STREAK_PLANS - 1 {
            let admitted = gate.admit(correcting_plan(), false, true);
            assert!(!admitted.is_correcting(), "streak not yet sustained");
        }
        let admitted = gate.admit(correcting_plan(), false, true);
        assert!(admitted.is_correcting(), "sustained streak must engage");
    }

    #[test]
    fn test_gate_streak_resets_on_idle_plan() {
        let mut gate = EngageGate::new();
        for _ in 0..ENGAGE_STREAK_PLANS - 1 {
            gate.admit(correcting_plan(), false, true);
        }
        gate.admit(CorrectionSchedule::default(), false, true);
        let admitted = gate.admit(correcting_plan(), false, true);
        assert!(
            !admitted.is_correcting(),
            "an idle plan must reset the streak"
        );
    }

    #[test]
    fn test_gate_cold_filter_never_engages() {
        let mut gate = EngageGate::new();
        for _ in 0..ENGAGE_STREAK_PLANS * 2 {
            let admitted = gate.admit(correcting_plan(), false, false);
            assert!(!admitted.is_correcting(), "cold filter must not engage");
        }
    }

    #[test]
    fn test_gate_reanchor_bypasses() {
        let mut gate = EngageGate::new();
        let reanchor = CorrectionSchedule {
            insert_every_n_frames: 0,
            drop_every_n_frames: 0,
            reanchor: true,
        };
        let admitted = gate.admit(reanchor, false, false);
        assert!(admitted.reanchor, "reanchor must bypass the gate");
    }

    #[test]
    fn test_gate_running_correction_replans_freely() {
        let mut gate = EngageGate::new();
        let admitted = gate.admit(correcting_plan(), true, true);
        assert!(
            admitted.is_correcting(),
            "a running correction must replan without gating"
        );
        let disengage = gate.admit(CorrectionSchedule::default(), true, true);
        assert!(
            !disengage.is_correcting(),
            "disengage must pass immediately"
        );
    }

    // -- Closed-loop stability --
    //
    // The planner, filter and gate are individually tested above, but the
    // property that matters in the field is what they do *together* against a
    // real plant. These tests close the loop: the correction the planner asks
    // for is applied to a simulated audio timeline, the resulting error is fed
    // back through the filter, and the loop runs for wall-clock minutes.
    //
    // The plant models the two forces that actually move the error:
    //   1. the drop/insert cadence this module schedules, and
    //   2. a constant clock-rate offset between the local sound card crystal
    //      and the server clock — an ordinary tens-of-ppm mismatch that no
    //      position-only controller can null out, so it re-accumulates error
    //      between episodes.
    //
    // Parameters are measured, not invented. On a PipeWire host (symmetra,
    // 2026-09-08) the process callback fires every 256.1ms (median of 66
    // inter-callback intervals derived from logged callback counters; a
    // 12288-frame quantum at 48kHz), and the idle-gap slope between correction
    // episodes implies a ~50ppm clock offset. Those two values reproduce the
    // field behaviour to within a few percent: a 148s limit cycle alternating
    // a 27s 0.18% slow episode with a 7s 1.27% fast episode.

    /// Measured PipeWire process-callback period (see module note above).
    const PIPEWIRE_PERIOD: f64 = 0.2561;
    /// Callback period the filter window was sized for ("~1s at the common
    /// 10ms WASAPI period", per `SYNC_ERROR_WINDOW`).
    const DESIGN_PERIOD: f64 = 0.010;
    /// Sound-card/server clock-rate mismatch, as a fraction (50ppm).
    const CLOCK_OFFSET: f64 = -50e-6;

    /// One engaged correction episode observed by the harness.
    #[derive(Debug, Clone, Copy)]
    struct Episode {
        start_s: f64,
        duration_s: f64,
        /// Speed change applied, as a percentage (always positive).
        speed_pct: f64,
        error_at_engage_us: f64,
        error_at_disengage_us: f64,
    }

    /// Run the real planner/filter/gate in closed loop over `secs` of
    /// simulated time and return every correction episode that engaged.
    fn run_closed_loop(period_s: f64, secs: f64, planner: &CorrectionPlanner) -> Vec<Episode> {
        const SAMPLE_RATE: u32 = 48_000;

        let mut filter = SyncErrorFilter::new();
        filter.set_callback_period((period_s * 1_000_000.0) as u64);
        let mut gate = EngageGate::new();
        let mut schedule = CorrectionSchedule::default();
        let mut error_us = 0.0f64;
        let mut t = 0.0f64;

        let mut episodes = Vec::new();
        let mut open: Option<(f64, f64, f64)> = None; // start_s, speed_pct, error at engage

        while t < secs {
            let filtered = filter.update(error_us.round() as i64);
            let planned = planner.plan(filtered, SAMPLE_RATE, schedule.is_correcting());
            schedule = gate.admit(planned, schedule.is_correcting(), filter.is_warm());

            // A reanchor is a silent cursor jump, not a rate change: model it
            // as the error being zeroed and the estimator restarted.
            if schedule.reanchor {
                error_us = 0.0;
                filter.reset();
                gate.reset();
                schedule = CorrectionSchedule::default();
            }

            // Inserting frames stretches the timeline (error rises toward
            // zero); dropping frames compresses it (error falls).
            let rate = if schedule.insert_every_n_frames > 0 {
                1.0 / f64::from(schedule.insert_every_n_frames)
            } else if schedule.drop_every_n_frames > 0 {
                -1.0 / f64::from(schedule.drop_every_n_frames)
            } else {
                0.0
            };

            match (open, rate != 0.0) {
                (None, true) => open = Some((t, rate.abs() * 100.0, error_us)),
                (Some((start_s, speed_pct, engage_err)), false) => {
                    episodes.push(Episode {
                        start_s,
                        duration_s: t - start_s,
                        speed_pct,
                        error_at_engage_us: engage_err,
                        error_at_disengage_us: error_us,
                    });
                    open = None;
                }
                _ => {}
            }

            error_us += (rate + CLOCK_OFFSET) * period_s * 1_000_000.0;
            t += period_s;
        }
        episodes
    }

    /// Corrections above ~1% are a clearly audible pitch shift (1% is a sixth
    /// of a semitone); below ~0.5% they are inaudible on program material.
    const AUDIBLE_SPEED_PCT: f64 = 1.0;

    #[test]
    fn test_closed_loop_is_quiet_at_the_design_callback_period() {
        // Sanity anchor for the harness: at the callback period the filter
        // window was sized for, the same loop against the same clock offset
        // stays inaudible. If this fails, the harness is wrong, not the code.
        let episodes = run_closed_loop(DESIGN_PERIOD, 1_200.0, &CorrectionPlanner::new());
        let worst = episodes.iter().map(|e| e.speed_pct).fold(0.0, f64::max);
        assert!(
            worst < AUDIBLE_SPEED_PCT,
            "at a {:.0}ms callback the loop should stay inaudible, but applied {:.2}%",
            DESIGN_PERIOD * 1000.0,
            worst,
        );
    }

    #[test]
    fn test_closed_loop_does_not_limit_cycle_at_pipewire_callback_period() {
        // Field regression: a 256ms callback stretches SYNC_ERROR_WINDOW from
        // the ~1s it was designed for to ~26s. The floor then holds a stale
        // minimum for that whole window, so an episode planned to clear 3.6ms
        // in `target_seconds` (2s) instead runs ~27s and injects ~46ms of the
        // opposite error — which the next episode must remove at an audible
        // rate. The loop never settles.
        let episodes = run_closed_loop(PIPEWIRE_PERIOD, 1_200.0, &CorrectionPlanner::new());
        let audible: Vec<_> = episodes
            .iter()
            .filter(|e| e.speed_pct >= AUDIBLE_SPEED_PCT)
            .collect();
        assert!(
            audible.is_empty(),
            "loop limit-cycles at a {:.0}ms callback: {} audible episodes in 1200s, \
             worst {:.2}% for {:.1}s (first at t={:.0}s)",
            PIPEWIRE_PERIOD * 1000.0,
            audible.len(),
            audible.iter().map(|e| e.speed_pct).fold(0.0, f64::max),
            audible.iter().map(|e| e.duration_s).fold(0.0, f64::max),
            audible.first().map(|e| e.start_s).unwrap_or(0.0),
        );
    }

    #[test]
    fn test_correction_episode_does_not_overshoot_past_its_target() {
        // An episode exists to remove the error it engaged on. Leaving a
        // larger error of the opposite sign means the corrector is the thing
        // creating the next episode's work.
        for &period in &[DESIGN_PERIOD, PIPEWIRE_PERIOD] {
            let episodes = run_closed_loop(period, 1_200.0, &CorrectionPlanner::new());
            for e in &episodes {
                // Some overshoot is unavoidable: the floor lags a rising error
                // by one window, and that dead time is inside the loop. What
                // must not happen is an episode leaving *grossly* more error
                // than it engaged on — that is the corrector manufacturing the
                // next episode's work, and it is what made the next episode
                // audible. One engage threshold of slop is the honest bound.
                let tolerated = e.error_at_engage_us.abs() + 3_000.0;
                assert!(
                    e.error_at_disengage_us.abs() <= tolerated,
                    "at a {:.0}ms callback, an episode engaged on {:.1}ms and \
                     disengaged on {:.1}ms after {:.1}s at {:.2}% — it overshot",
                    period * 1000.0,
                    e.error_at_engage_us / 1000.0,
                    e.error_at_disengage_us / 1000.0,
                    e.duration_s,
                    e.speed_pct,
                );
            }
        }
    }

    #[test]
    fn test_window_span_is_inactive_at_classic_callback_periods() {
        // The bound must not disturb backends that already work: cpal/WASAPI
        // periods keep the full sample window they were tuned with.
        for period_ms in [5, 10, 20] {
            assert_eq!(
                window_len_for_period(period_ms * 1_000),
                SYNC_ERROR_WINDOW,
                "{period_ms}ms callback should keep the full window"
            );
        }
        // 40ms is the longest classic period; 4s/40ms = 100 of 101 samples.
        assert_eq!(window_len_for_period(40_000), 100);
    }

    #[test]
    fn test_window_shrinks_to_hold_the_span_at_long_callback_periods() {
        // PipeWire handing the stream a 12288-frame quantum at 48kHz.
        assert_eq!(window_len_for_period(256_100), 16);
        let span_s = 16.0 * 0.2561;
        assert!(
            span_s <= 4.5,
            "window should cover ~4s of wall time, covers {span_s:.1}s"
        );
    }

    #[test]
    fn test_window_never_shrinks_below_the_floor_minimum() {
        // A floor over a couple of samples is one outlier away from useless.
        assert_eq!(window_len_for_period(10_000_000), MIN_SYNC_ERROR_WINDOW);
        assert_eq!(window_len_for_period(u64::MAX), MIN_SYNC_ERROR_WINDOW);
    }

    #[test]
    fn test_unknown_callback_period_keeps_the_full_window() {
        assert_eq!(window_len_for_period(0), SYNC_ERROR_WINDOW);
    }

    #[test]
    fn test_callback_period_change_discards_stale_history() {
        // Samples taken at a different wake cadence describe a different
        // window, so a period change must cool the filter rather than let a
        // stale floor decide the next engage.
        let mut filter = SyncErrorFilter::new();
        filter.set_callback_period(10_000);
        for _ in 0..SYNC_ERROR_WINDOW {
            filter.update(5_000);
        }
        assert!(
            filter.is_warm(),
            "filter should be warm after a full window"
        );

        filter.set_callback_period(256_100);
        assert!(!filter.is_warm(), "a period change must cool the filter");

        // ...and warm again on the new, shorter window.
        for _ in 0..window_len_for_period(256_100) {
            filter.update(5_000);
        }
        assert!(filter.is_warm());
    }

    #[test]
    fn test_repeated_identical_period_does_not_cool_the_filter() {
        // The audio callback calls this every wake; it must be idempotent.
        let mut filter = SyncErrorFilter::new();
        filter.set_callback_period(256_100);
        for _ in 0..window_len_for_period(256_100) {
            filter.update(1_000);
        }
        assert!(filter.is_warm());
        filter.set_callback_period(256_100);
        assert!(filter.is_warm(), "same period must not reset the window");
    }

    #[test]
    fn test_closed_loop_stays_inaudible_across_the_callback_period_range() {
        // Backends hand out very different quanta; none of them may push the
        // loop into an audible limit cycle.
        for period_ms in [5.0, 10.0, 20.0, 40.0, 64.0, 128.0, 256.1, 512.0] {
            let period_s = period_ms / 1000.0;
            let episodes = run_closed_loop(period_s, 2_400.0, &CorrectionPlanner::new());
            let worst = episodes.iter().map(|e| e.speed_pct).fold(0.0, f64::max);
            assert!(
                worst < AUDIBLE_SPEED_PCT,
                "at a {period_ms}ms callback the loop applied {worst:.2}%"
            );
        }
    }
}
