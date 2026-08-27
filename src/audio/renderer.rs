// ABOUTME: Backend-agnostic audio renderer with drift correction
// ABOUTME: Shared by cpal and PipeWire backends for synced playback

use crate::audio::gain::{GainControl, GainRamp};
use crate::audio::sync_correction::{
    CorrectionPlanner, CorrectionSchedule, EngageGate, SyncErrorFilter,
};
use crate::audio::{AudioBuffer, AudioFormat};
use crate::log_sampling::should_log_sample;
use crate::sync::ClockSync;
use cpal::Sample;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Callback for post-processing audio samples before output.
///
/// Receives `&mut [f32]` (interleaved, after gain is applied).
///
/// The callback is invoked on **every** audio callback, including during
/// pre-start silence when the buffer is all zeros. This allows consumers
/// (e.g. VU meters) to observe the silence rather than missing callbacks.
///
/// # Thread Safety
///
/// This closure runs on the **audio callback thread**. It must:
/// - Not block (no locks, I/O, or sleeping)
/// - Not allocate (no `Vec::push`, `Box::new`, etc.)
/// - Not panic (would abort the audio thread)
///
/// # Why `Box<dyn>`?
///
/// Using dynamic dispatch (`Box<dyn FnMut>`) keeps players concrete,
/// non-generic types. This simplifies storage, trait object compatibility, and
/// downstream usage at the cost of one vtable indirect call per audio callback
/// (~1 ns vs the ~200 us callback budget).
pub type ProcessCallback = Box<dyn FnMut(&mut [f32]) + Send + 'static>;

pub(crate) struct PlaybackQueue {
    pub(crate) queue: VecDeque<AudioBuffer>,
    pub(crate) current: Option<AudioBuffer>,
    pub(crate) index: usize,
    /// Current playback position in **server-time microseconds**. Periodically
    /// reanchored to the server's clock during clock-sync correction, so this
    /// represents "what server timestamp is playing right now", not how much
    /// audio content has been consumed.
    pub(crate) cursor_us: i64,
    pub(crate) cursor_remainder: i64,
    pub(crate) initialized: bool,
    pub(crate) generation: u64,
    pub(crate) force_reanchor: bool,
    /// Buffers enqueued in this generation; the sampling key for the enqueue
    /// trace line. Reset by `clear()`.
    pub(crate) enqueue_count: u64,
}

impl PlaybackQueue {
    pub(crate) fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            current: None,
            index: 0,
            cursor_us: 0,
            cursor_remainder: 0,
            initialized: false,
            generation: 0,
            force_reanchor: true,
            enqueue_count: 0,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.queue.clear();
        self.current = None;
        self.index = 0;
        self.cursor_us = 0;
        self.cursor_remainder = 0;
        self.initialized = false;
        self.generation = self.generation.wrapping_add(1);
        self.force_reanchor = true;
        self.enqueue_count = 0;
    }

    pub(crate) fn push(&mut self, buffer: AudioBuffer) {
        // Initialize the cursor from the first enqueued buffer so the audio
        // callback can see a valid cursor_us before it starts reading. Without
        // this, the callback's pre-start gate can't evaluate timestamps and
        // outputs silence indefinitely. Use the minimum timestamp seen so far
        // since buffers may arrive out of order.
        if !self.initialized {
            self.cursor_us = buffer.timestamp;
            self.cursor_remainder = 0;
            self.initialized = true;
        }

        // When the server rebases its timeline backward (e.g. after event loop
        // starvation), new chunks arrive with timestamps that overlap chunks
        // already in the queue. Remove all overlapping buffers to prevent
        // duplicate audio that causes audible stuttering (~500ms of audio
        // played twice). The server will send fresh buffers for any gaps.
        // This works for any chunk size (the sendspin spec allows arbitrarily
        // small chunks), unlike the previous fixed-threshold approach.
        //
        // The overlap must be at least one frame to count. A sub-frame
        // "overlap" cannot contain a duplicated sample — it is timestamp
        // rounding, not a rebase: at rates where chunks are not a whole
        // number of microseconds (44.1kHz: 1102 frames = 24988.66µs), the
        // server's floor-based timestamp grid steps 24988µs while we measure
        // the buffer as 24989µs, landing a third of all chunks 1µs "inside"
        // their predecessor. Evicting on those phantom overlaps silently
        // discarded ~34% of all 44.1kHz audio (heard as continuous popping).
        let rate = i64::from(buffer.format.sample_rate.max(1));
        let frame_us = (1_000_000 + rate - 1) / rate;
        let new_end = buffer.timestamp + buffer.duration_us();
        self.queue.retain(|b| {
            let existing_end = b.timestamp + b.duration_us();
            let overlap_us = new_end.min(existing_end) - buffer.timestamp.max(b.timestamp);
            overlap_us < frame_us
        });

        let pos = self
            .queue
            .iter()
            .position(|b| b.timestamp > buffer.timestamp);
        if let Some(pos) = pos {
            self.queue.insert(pos, buffer);
        } else {
            self.queue.push_back(buffer);
        }
    }

    pub(crate) fn next_frame(&mut self, channels: usize, sample_rate: u32) -> Option<&[i32]> {
        let needs_buffer = match self.current {
            None => true,
            Some(ref c) => self.index + channels > c.samples.len(),
        };
        if needs_buffer {
            // Drop stale buffers that are entirely before the cursor.
            if self.initialized {
                while let Some(front) = self.queue.front() {
                    if front.timestamp + front.duration_us() < self.cursor_us {
                        let _ = self.queue.pop_front();
                        continue;
                    }
                    break;
                }
            }

            // Pop buffers until we find one with remaining samples past the
            // cursor, or the queue is empty.
            loop {
                self.current = self.queue.pop_front();
                self.index = 0;

                // Skip past samples that are behind the cursor. This handles
                // buffers that partially overlap with the current playback
                // position, e.g. from backward timestamp jumps during server
                // timeline rebases. Without this, playing from the start of
                // such a buffer repeats audio the cursor has already passed,
                // causing an audible stutter.
                if self.initialized {
                    if let Some(ref current) = self.current {
                        if current.timestamp < self.cursor_us {
                            let skip_us = self.cursor_us - current.timestamp;
                            let skip_frames =
                                (skip_us.saturating_mul(sample_rate as i64) / 1_000_000) as usize;
                            if skip_frames > 0 {
                                self.index = skip_frames
                                    .saturating_mul(channels)
                                    .min(current.samples.len());
                            }
                        }
                    }
                }

                // If the skip consumed the entire buffer (or left fewer
                // samples than one frame), discard it and try the next one.
                match self.current {
                    Some(ref c) if self.index + channels > c.samples.len() => {
                        self.current = None;
                        if self.queue.is_empty() {
                            break;
                        }
                        continue;
                    }
                    _ => break,
                }
            }
        }

        if !self.initialized {
            if let Some(current) = self.current.as_ref() {
                self.cursor_us = current.timestamp;
                self.cursor_remainder = 0;
                self.initialized = true;
            }
        }

        // Bail before advancing cursor/index when the queue is empty.
        // Without this the cursor races ahead during underruns, causing
        // the stale-buffer-dropping logic to discard valid buffers when
        // audio resumes.
        self.current.as_ref()?;

        let start = self.index;
        let end = self.index + channels;
        self.index = end;
        self.advance_cursor(sample_rate);

        Some(&self.current.as_ref()?.samples[start..end])
    }

    fn advance_cursor(&mut self, sample_rate: u32) {
        self.cursor_remainder += 1_000_000;
        let advance = self.cursor_remainder / sample_rate as i64;
        self.cursor_remainder %= sample_rate as i64;
        self.cursor_us += advance;
    }

    pub(crate) fn first_playable_cursor_at_or_after(&self, server_time_us: i64) -> Option<i64> {
        if let Some(buffer) = self.current.as_ref() {
            let remaining_start = buffer.timestamp.max(self.cursor_us);
            if buffer.timestamp + buffer.duration_us() > server_time_us.max(remaining_start) {
                return Some(remaining_start.max(server_time_us));
            }
        }

        for buffer in &self.queue {
            if buffer.timestamp + buffer.duration_us() > server_time_us {
                return Some(buffer.timestamp.max(server_time_us));
            }
        }

        None
    }

    pub(crate) fn queued_frames(&self, channels: usize) -> usize {
        let current_frames = self.current.as_ref().map_or(0, |current| {
            current.samples.len().saturating_sub(self.index) / channels
        });
        let queued_frames = self
            .queue
            .iter()
            .map(|buffer| buffer.samples.len() / channels)
            .sum::<usize>();
        current_frames + queued_frames
    }

    pub(crate) fn queued_duration_us(&self, channels: usize, sample_rate: u32) -> u64 {
        self.queued_frames(channels) as u64 * 1_000_000 / sample_rate.max(1) as u64
    }

    pub(crate) fn buffer_count(&self) -> usize {
        self.queue.len() + usize::from(self.current.is_some())
    }
}

/// Microseconds as fractional milliseconds, for log formatting.
pub(crate) fn us_to_ms(us: u64) -> f64 {
    us as f64 / 1000.0
}

/// Queue depth below which the edge-triggered "queue low" debug line fires.
const QUEUE_LOW_WATER_US: u64 = 100_000;

/// Queue depth required to log recovery after a low-queue warning. Kept above
/// [`QUEUE_LOW_WATER_US`] so a queue hovering at one boundary cannot flood the
/// log with low/recovered pairs.
const QUEUE_RECOVERED_WATER_US: u64 = 200_000;

/// Diagnostic counters for the audio callback. Logging-only: playback
/// decisions never read these.
///
/// `callbacks` counts for the lifetime of the stream and anchors every log
/// line to one timeline. The rest are per-generation — reset whenever the
/// playback queue generation changes — so each stream start reports its own
/// startup behavior.
#[derive(Default)]
struct CallbackStats {
    /// Data callbacks since the stream was built. Never reset.
    callbacks: u64,
    /// Callbacks that emitted silence (pre-start gate, reanchor wait, early).
    silent_callbacks: u64,
    /// Callbacks that skipped sync because the clock lock was contended.
    sync_lock_misses: u64,
    /// Frames filled with silence because the queue ran dry.
    underrun_frames: u64,
    /// Callbacks that had at least one underrun frame.
    underrun_callbacks: u64,
    /// Length of the current run of underrun callbacks (0 while healthy).
    consecutive_underrun_callbacks: u64,
    /// Schedule updates within the current correction episode; the sampling
    /// key for the correction trace line. Reset when correction disengages.
    correction_updates: u64,
    /// Corrections the planner requested during clock warm-up that were
    /// suppressed; the sampling key for the warm-up trace line.
    warmup_suppressed_corrections: u64,
    /// Corrections the planner requested that the engage gate suppressed
    /// while awaiting a sustained error; the sampling key for its trace line.
    gate_suppressed_corrections: u64,
    /// Correction episodes started (idle -> correcting transitions, including
    /// reanchor engagements). Mirrors the "Sync correction engaged" debug
    /// line 1:1 so the generation summary can answer whether the corrector
    /// ever fired, even when debug logging was off during playback.
    correction_engagements: u64,
    /// Whether the queue was below [`QUEUE_LOW_WATER_US`] at the last render.
    /// Drives the edge-triggered low/recovered debug lines.
    queue_low: bool,
}

impl CallbackStats {
    /// Reset per-generation counters, keeping the lifetime callback count.
    fn reset_for_generation(&mut self) {
        *self = Self {
            callbacks: self.callbacks,
            ..Self::default()
        };
    }
}

/// Backend-agnostic audio renderer with drift correction.
///
/// Owns all per-stream callback state (correction planner/filters, stats,
/// gain ramp) that upstream keeps as locals captured by the cpal callback
/// closure. Both backends call [`AudioRenderer::render`] from their audio
/// callback with a caller-supplied notion of output latency:
/// - **cpal**: `playback_delta = ts.playback - ts.callback` from
///   `OutputCallbackInfo`
/// - **PipeWire**: the sink delay reported by `pw_stream_get_time_n`
pub(crate) struct AudioRenderer {
    queue: Arc<Mutex<PlaybackQueue>>,
    clock_sync: Arc<Mutex<ClockSync>>,
    channels: usize,
    sample_rate: u32,
    gain_control: GainControl,
    process_callback: Option<ProcessCallback>,
    /// Shared with the owning player; see `SyncedPlayer::set_static_delay`.
    static_delay_us: Arc<AtomicU64>,
    planner: CorrectionPlanner,
    error_filter: SyncErrorFilter,
    engage_gate: EngageGate,
    last_frame: Vec<i32>,
    schedule: CorrectionSchedule,
    insert_counter: u32,
    drop_counter: u32,
    started: bool,
    handoff_warned: bool,
    sync_settle_logged: bool,
    last_callback_instant: Option<Instant>,
    last_playback_delta_us: Option<u64>,
    // Running minimum of the presentation-latency snapshot, reset per
    // generation. Reanchors anchor against this floor rather than one
    // wake's reading: padding noise is one-sided (see SyncErrorFilter),
    // so a single sample may run a whole period high, and anchoring to it
    // bakes that period into the timeline until corrections audibly
    // unwind it. A stale floor after a latency-regime shift costs at most
    // one period of realignment — no worse than the shift itself.
    min_playback_delta_us: u64,
    last_generation: u64,
    stats: CallbackStats,
    gain_ramp: GainRamp,
}

impl AudioRenderer {
    pub(crate) fn new(
        queue: Arc<Mutex<PlaybackQueue>>,
        clock_sync: Arc<Mutex<ClockSync>>,
        format: &AudioFormat,
        gain_control: GainControl,
        process_callback: Option<ProcessCallback>,
        static_delay_us: Arc<AtomicU64>,
    ) -> Self {
        let initial_gain = gain_control.gain();
        Self {
            queue,
            clock_sync,
            channels: format.channels as usize,
            sample_rate: format.sample_rate,
            gain_control,
            process_callback,
            static_delay_us,
            planner: CorrectionPlanner::new(),
            error_filter: SyncErrorFilter::new(),
            engage_gate: EngageGate::new(),
            last_frame: vec![i32::EQUILIBRIUM; format.channels as usize],
            schedule: CorrectionSchedule::default(),
            insert_counter: 0,
            drop_counter: 0,
            started: false,
            handoff_warned: false,
            sync_settle_logged: false,
            last_callback_instant: None,
            last_playback_delta_us: None,
            min_playback_delta_us: u64::MAX,
            last_generation: 0,
            stats: CallbackStats::default(),
            gain_ramp: GainRamp::new(format.sample_rate, initial_gain),
        }
    }

    /// Render one callback's worth of interleaved f32 frames into `data`.
    ///
    /// `callback_instant` is when the audio callback fired; `playback_delta`
    /// is how far in the future these samples will physically play
    /// (presentation latency). Fills `data` completely on every call
    /// (silence while gated), applies gain, and invokes the process
    /// callback last.
    pub(crate) fn render(
        &mut self,
        data: &mut [f32],
        callback_instant: Instant,
        playback_delta: Duration,
    ) {
        let AudioRenderer {
            queue,
            clock_sync,
            channels,
            sample_rate,
            gain_control,
            process_callback,
            static_delay_us,
            planner,
            error_filter,
            engage_gate,
            last_frame,
            schedule,
            insert_counter,
            drop_counter,
            started,
            handoff_warned,
            sync_settle_logged,
            last_callback_instant,
            last_playback_delta_us,
            min_playback_delta_us,
            last_generation,
            stats,
            gain_ramp,
        } = self;
        let channels = *channels;
        let sample_rate = *sample_rate;

        // Advance the gain ramp even while silent so the first real
        // audio resumes at the target gain with no fade-in.
        let mut emit_silence = |data: &mut [f32]| {
            let target = gain_control.gain();
            gain_ramp.advance(data.len() / channels, target);
            data.fill(0.0);
            if let Some(cb) = process_callback.as_mut() {
                cb(data);
            }
        };

        // Snapshot the level checks once per callback. At info level
        // these two loads are the only per-callback logging cost.
        let debug_logging = log::log_enabled!(log::Level::Debug);
        let trace_logging = log::log_enabled!(log::Level::Trace);

        stats.callbacks += 1;
        let frames = data.len() / channels;

        // Read queue timing state together. The generation is
        // rechecked before consuming force_reanchor so a clear()
        // racing with this callback cannot clear the next startup's
        // one-shot handoff.
        let (generation, cursor_us, force_reanchor, queued_us, queued_buffers) = {
            let queue = queue.lock();
            let cursor = if queue.initialized {
                Some(queue.cursor_us)
            } else {
                None
            };
            // Queue depth costs a walk over every queued buffer,
            // so only measure it when a log line below can print
            // it.
            let (queued_us, queued_buffers) = if debug_logging {
                (
                    queue.queued_duration_us(channels, sample_rate),
                    queue.buffer_count(),
                )
            } else {
                (0, 0)
            };
            (
                queue.generation,
                cursor,
                queue.force_reanchor,
                queued_us,
                queued_buffers,
            )
        };
        if generation != *last_generation {
            log::debug!(
                            "Playback queue generation changed: {} -> {}, queued={:.1}ms, buffers={}, callbacks={}, silent_callbacks={}, underrun_callbacks={}, underrun_frames={}, sync_lock_misses={}, correction_engagements={}",
                            last_generation,
                            generation,
                            us_to_ms(queued_us),
                            queued_buffers,
                            stats.callbacks,
                            stats.silent_callbacks,
                            stats.underrun_callbacks,
                            stats.underrun_frames,
                            stats.sync_lock_misses,
                            stats.correction_engagements,
                        );
            *last_generation = generation;
            *started = false;
            *schedule = CorrectionSchedule::default();
            *insert_counter = 0;
            *drop_counter = 0;
            error_filter.reset();
            engage_gate.reset();
            *min_playback_delta_us = u64::MAX;
            stats.reset_for_generation();
            for sample in last_frame.iter_mut() {
                *sample = i32::EQUILIBRIUM;
            }
            *handoff_warned = false;
        }

        let playback_instant = callback_instant + playback_delta;

        // Both values are normally steady, so a step in either
        // explains a sync-error step: a callback gap means this
        // thread stalled; a playback-delta shift means the OS
        // moved the presentation timeline.
        let playback_delta_us = playback_delta.as_micros() as u64;
        if let Some(last) = *last_callback_instant {
            let gap_us = callback_instant.duration_since(last).as_micros() as u64;
            let period_us = frames as u64 * 1_000_000 / u64::from(sample_rate.max(1));
            if gap_us >= 2 * period_us {
                log::debug!(
                                "Audio callback gap: {:.1}ms since previous (period ~{:.1}ms), callback={}, generation={}",
                                us_to_ms(gap_us),
                                us_to_ms(period_us),
                                stats.callbacks,
                                generation,
                            );
            }
        }
        *last_callback_instant = Some(callback_instant);
        if let Some(last) = *last_playback_delta_us {
            if playback_delta_us.abs_diff(last) > 1_000 {
                log::debug!(
                                "Output timeline shifted: playback_delta {:.1}ms -> {:.1}ms, callback={}, generation={}",
                                us_to_ms(last),
                                us_to_ms(playback_delta_us),
                                stats.callbacks,
                                generation,
                            );
            }
        }
        *last_playback_delta_us = Some(playback_delta_us);
        *min_playback_delta_us = (*min_playback_delta_us).min(playback_delta_us);

        // try_lock: skip sync if contended rather than blocking
        // the audio thread. force_reanchor is sticky in the
        // queue, so it will be retried on the next callback.
        let sync = clock_sync.try_lock();
        if cursor_us.is_some() && sync.is_none() {
            // Count lock contention only once playback has an
            // initialized cursor; before that there is no timeline
            // position to synchronize yet.
            stats.sync_lock_misses += 1;
            if trace_logging && should_log_sample(stats.sync_lock_misses) {
                log::trace!(
                                "Audio callback skipped sync: clock lock contended, callback={}, sync_lock_miss={}, queued={:.1}ms, buffers={}, started={}",
                                stats.callbacks,
                                stats.sync_lock_misses,
                                us_to_ms(queued_us),
                                queued_buffers,
                                started,
                            );
            }
        }
        if let (Some(cursor_us), Some(sync)) = (cursor_us, sync) {
            // Emit each sample `delay` earlier so downstream
            // (amp/speaker) latency lands it on time. The reanchor
            // below adds the same delay in the local→server
            // direction; the two signs must stay in step or the
            // planner chases a phantom error every callback.
            let delay_us = static_delay_us.load(Ordering::Relaxed);
            let mut effective_cursor_us = cursor_us;
            let sync_settled = sync.is_settled();
            if sync_settled && !*sync_settle_logged {
                *sync_settle_logged = true;
                // Warm-up measurements track the converging clock
                // estimate, not playback; start the filter fresh.
                error_filter.reset();
                engage_gate.reset();
                log::debug!(
                                "Clock sync settled; corrections enabled: callback={}, suppressed_during_warmup={}, generation={}",
                                stats.callbacks,
                                stats.warmup_suppressed_corrections,
                                generation,
                            );
            }

            if force_reanchor {
                let mut reanchor_applied = false;
                // Anchor from the delta floor, not this wake's
                // reading (see min_playback_delta_us above). The
                // floor already includes this callback's sample,
                // so it is never u64::MAX here.
                let anchor_instant =
                    callback_instant + Duration::from_micros(*min_playback_delta_us);
                let handoff_instant = if *started {
                    anchor_instant
                } else {
                    // Startup handoff: anchor to `+ handoff_delta` (this buffer's
                    // end = the next callback's start) so the next start gate sees
                    // `expected ≈ playback_instant`. Playing now would misalign the
                    // cursor by one buffer, so we stay silent for this one period.
                    let handoff_delta = Duration::from_secs_f64(frames as f64 / sample_rate as f64);
                    anchor_instant + handoff_delta
                };
                let client_micros =
                    sync.instant_to_client_micros(handoff_instant) + delay_us as i64;
                if let Some(server_time) = sync.client_to_server_micros(client_micros) {
                    let mut queue = queue.lock();
                    if queue.generation == generation && queue.initialized {
                        if let Some(cursor_us) =
                            queue.first_playable_cursor_at_or_after(server_time)
                        {
                            queue.cursor_us = cursor_us;
                            queue.cursor_remainder = 0;
                            queue.force_reanchor = false;
                            effective_cursor_us = cursor_us;
                            reanchor_applied = true;
                            *schedule = CorrectionSchedule::default();
                            *insert_counter = 0;
                            *drop_counter = 0;
                            // The cursor just jumped (e.g. a
                            // static-delay change, which does not
                            // bump the generation); prior
                            // measurements describe the old
                            // timeline.
                            error_filter.reset();
                            engage_gate.reset();
                            log::debug!(
                                "Sync reanchor applied: cursor reset to server_time={cursor_us}µs"
                            );
                        } else if !*handoff_warned {
                            *handoff_warned = true;
                            log::warn!(
                                "Sync reanchor: no playable buffer at or after \
                                             server_time={server_time}µs — staying silent"
                            );
                        }
                    }
                }

                if !reanchor_applied || !*started {
                    stats.silent_callbacks += 1;
                    if trace_logging && should_log_sample(stats.silent_callbacks) {
                        log::trace!(
                                        "Audio callback silent during reanchor: callback={}, silent_callback={}, reanchor_applied={}, started={}, queued={:.1}ms, buffers={}, generation={}",
                                        stats.callbacks,
                                        stats.silent_callbacks,
                                        reanchor_applied,
                                        started,
                                        us_to_ms(queued_us),
                                        queued_buffers,
                                        generation,
                                    );
                    }
                    emit_silence(data);
                    return;
                }
            }

            if let Some(expected_instant) =
                sync.server_to_local_instant_with_latency(effective_cursor_us, delay_us)
            {
                // Pre-start only: hold silence until the cursor's
                // scheduled instant. After start, "early" readings
                // are jitter — injecting silence here caused real
                // dropouts (audible blips); the planner handles
                // sustained earliness instead.
                let early_window = Duration::from_millis(1);
                if !*started && playback_instant + early_window < expected_instant {
                    stats.silent_callbacks += 1;
                    if trace_logging && should_log_sample(stats.silent_callbacks) {
                        let early_us = expected_instant
                            .duration_since(playback_instant)
                            .as_micros() as u64;
                        log::trace!(
                                        "Audio callback early; emitting silence: callback={}, silent_callback={}, early={:.1}ms, cursor={}µs, queued={:.1}ms, buffers={}, generation={}",
                                        stats.callbacks,
                                        stats.silent_callbacks,
                                        us_to_ms(early_us),
                                        effective_cursor_us,
                                        us_to_ms(queued_us),
                                        queued_buffers,
                                        generation,
                                    );
                    }
                    emit_silence(data);
                    return;
                }
                if !*started {
                    *started = true;
                    log::debug!(
                                    "Audio playback started: callback={}, cursor={}µs, queued={:.1}ms, buffers={}, silent_callbacks_before_start={}, sync_lock_misses={}",
                                    stats.callbacks,
                                    effective_cursor_us,
                                    us_to_ms(queued_us),
                                    queued_buffers,
                                    stats.silent_callbacks,
                                    stats.sync_lock_misses,
                                );
                }

                let raw_error_us = if playback_instant >= expected_instant {
                    playback_instant
                        .duration_since(expected_instant)
                        .as_micros() as i64
                } else {
                    -(expected_instant
                        .duration_since(playback_instant)
                        .as_micros() as i64)
                };
                // A single reading can sit a whole engine period
                // above true alignment while the FIFO plays
                // gaplessly (see SyncErrorFilter); plan against
                // the window floor, never one wake's snapshot.
                let error_us = error_filter.update(raw_error_us);
                let planned_schedule =
                    planner.plan(error_us, sample_rate, schedule.is_correcting());
                // Corrections mutate audible frames: engage only
                // on sustained evidence over a warm filter (see
                // EngageGate).
                let gated_schedule = engage_gate.admit(
                    planned_schedule,
                    schedule.is_correcting(),
                    error_filter.is_warm(),
                );
                if gated_schedule != planned_schedule {
                    stats.gate_suppressed_corrections += 1;
                    if trace_logging && should_log_sample(stats.gate_suppressed_corrections) {
                        log::trace!(
                                        "Sync correction awaiting sustained error: callback={}, suppressed={}, error={:.3}ms, raw_error={:.3}ms, generation={}",
                                        stats.callbacks,
                                        stats.gate_suppressed_corrections,
                                        error_us as f64 / 1000.0,
                                        raw_error_us as f64 / 1000.0,
                                        generation,
                                    );
                    }
                }
                let planned_schedule = gated_schedule;
                // While the sync estimate is still converging,
                // measured error is mostly movement of the
                // estimate itself; correcting for it chases
                // filter noise audibly. Trust the server's audio
                // until settled, honoring only gross reanchors.
                let new_schedule = if sync_settled || planned_schedule.reanchor {
                    planned_schedule
                } else {
                    if planned_schedule.is_correcting() {
                        stats.warmup_suppressed_corrections += 1;
                        if trace_logging && should_log_sample(stats.warmup_suppressed_corrections) {
                            log::trace!(
                                            "Sync correction suppressed during clock warm-up: callback={}, suppressed={}, error={:.3}ms, raw_error={:.3}ms, generation={}",
                                            stats.callbacks,
                                            stats.warmup_suppressed_corrections,
                                            error_us as f64 / 1000.0,
                                            raw_error_us as f64 / 1000.0,
                                            generation,
                                        );
                        }
                    }
                    CorrectionSchedule::default()
                };
                if new_schedule != *schedule {
                    if new_schedule.is_correcting() != schedule.is_correcting() {
                        if new_schedule.is_correcting() {
                            stats.correction_engagements += 1;
                            log::debug!(
                                            "Sync correction engaged: error={:.3}ms, raw_error={:.3}ms, insert_every={}, drop_every={}, reanchor={}, callback={}, generation={}",
                                            error_us as f64 / 1000.0,
                                            raw_error_us as f64 / 1000.0,
                                            new_schedule.insert_every_n_frames,
                                            new_schedule.drop_every_n_frames,
                                            new_schedule.reanchor,
                                            stats.callbacks,
                                            generation,
                                        );
                        } else {
                            // The floor lags rises, so error= may
                            // read worse than raw_error= here;
                            // expected, not a bug.
                            log::debug!(
                                            "Sync correction disengaged: error={:.3}ms, raw_error={:.3}ms, callback={}, generation={}",
                                            error_us as f64 / 1000.0,
                                            raw_error_us as f64 / 1000.0,
                                            stats.callbacks,
                                            generation,
                                        );
                        }
                    }
                    if new_schedule.is_correcting() {
                        // The cadence is re-planned as the error
                        // converges, which can change the schedule
                        // on every callback. Sample the updates so
                        // each correction episode logs its first
                        // few adjustments and then a heartbeat;
                        // engage/disengage transitions are logged
                        // at debug above and reanchor execution is
                        // logged where it is applied below.
                        stats.correction_updates += 1;
                        if trace_logging && should_log_sample(stats.correction_updates) {
                            log::trace!(
                                            "Sync correction updated: callback={}, correction_update={}, error={:.3}ms, raw_error={:.3}ms, insert_every={}, drop_every={}, reanchor={}, queued={:.1}ms, generation={}",
                                            stats.callbacks,
                                            stats.correction_updates,
                                            error_us as f64 / 1000.0,
                                            raw_error_us as f64 / 1000.0,
                                            new_schedule.insert_every_n_frames,
                                            new_schedule.drop_every_n_frames,
                                            new_schedule.reanchor,
                                            us_to_ms(queued_us),
                                            generation,
                                        );
                        }
                    } else {
                        stats.correction_updates = 0;
                    }
                    *schedule = new_schedule;
                    *insert_counter = schedule.insert_every_n_frames;
                    *drop_counter = schedule.drop_every_n_frames;
                }

                if schedule.reanchor {
                    // Mirror of the start-gate subtraction: audio
                    // emitted now is heard `delay_us` later, so
                    // anchor the cursor to that hear-instant —
                    // derived from the delta floor, not this
                    // wake's reading (see min_playback_delta_us).
                    let anchor_instant =
                        callback_instant + Duration::from_micros(*min_playback_delta_us);
                    let client_micros =
                        sync.instant_to_client_micros(anchor_instant) + delay_us as i64;
                    if let Some(server_time) = sync.client_to_server_micros(client_micros) {
                        let mut queue = queue.lock();
                        queue.cursor_us = server_time;
                        queue.cursor_remainder = 0;
                        log::debug!(
                            "Sync reanchor applied: cursor reset to server_time={server_time}µs"
                        );
                    }
                    *schedule = CorrectionSchedule::default();
                    *insert_counter = 0;
                    *drop_counter = 0;
                    stats.correction_updates = 0;
                    // The cursor just jumped; prior measurements
                    // describe the old timeline.
                    error_filter.reset();
                    engage_gate.reset();
                }
            } else if schedule.is_correcting() {
                // Conversions went dark (the implausible-drift
                // safety net): stop correcting rather than
                // resample blind on the stale cadence.
                log::debug!(
                                "Sync conversions unavailable; clearing correction schedule: callback={}, generation={}",
                                stats.callbacks,
                                generation,
                            );
                *schedule = CorrectionSchedule::default();
                *insert_counter = 0;
                *drop_counter = 0;
                stats.correction_updates = 0;
                error_filter.reset();
                engage_gate.reset();
            }
        }

        // If playback hasn't started yet (clock sync not converged,
        // lock contention, or pre-start gate active), output silence.
        // Audio data stays in the ring buffer for when sync converges
        // and reanchor positions the cursor correctly.
        if !*started {
            stats.silent_callbacks += 1;
            if trace_logging && should_log_sample(stats.silent_callbacks) {
                log::trace!(
                                "Audio callback silent before start: callback={}, silent_callback={}, cursor_present={}, queued={:.1}ms, buffers={}, generation={}",
                                stats.callbacks,
                                stats.silent_callbacks,
                                cursor_us.is_some(),
                                us_to_ms(queued_us),
                                queued_buffers,
                                generation,
                            );
            }
            emit_silence(data);
            return;
        }

        let (callback_underrun_frames, queued_after_us, buffers_after) = {
            let mut queue = queue.lock();
            let mut missing_frames = 0u64;
            let mut out_index = 0;

            for _ in 0..frames {
                if schedule.drop_every_n_frames > 0 {
                    *drop_counter = drop_counter.saturating_sub(1);
                    if *drop_counter == 0 {
                        // Discard one frame to catch up
                        let _ = queue.next_frame(channels, sample_rate);
                        *drop_counter = schedule.drop_every_n_frames;
                        // Get and output the next frame (don't repeat last_frame)
                        if let Some(frame) = queue.next_frame(channels, sample_rate) {
                            last_frame.copy_from_slice(frame);
                            for sample in frame {
                                data[out_index] = f32::from_sample(*sample);
                                out_index += 1;
                            }
                        } else {
                            for sample in last_frame.iter() {
                                data[out_index] = f32::from_sample(*sample);
                                out_index += 1;
                            }
                        }
                        continue;
                    }
                }

                if schedule.insert_every_n_frames > 0 {
                    *insert_counter = insert_counter.saturating_sub(1);
                    if *insert_counter == 0 {
                        *insert_counter = schedule.insert_every_n_frames;
                        for sample in last_frame.iter() {
                            data[out_index] = f32::from_sample(*sample);
                            out_index += 1;
                        }
                        continue;
                    }
                }

                if let Some(frame) = queue.next_frame(channels, sample_rate) {
                    last_frame.copy_from_slice(frame);
                    for sample in frame {
                        data[out_index] = f32::from_sample(*sample);
                        out_index += 1;
                    }
                } else {
                    missing_frames += 1;
                    for _ in 0..channels {
                        data[out_index] = 0.0;
                        out_index += 1;
                    }
                }
            }

            let (queued_after_us, buffers_after) = if debug_logging {
                (
                    queue.queued_duration_us(channels, sample_rate),
                    queue.buffer_count(),
                )
            } else {
                (0, 0)
            };
            (missing_frames, queued_after_us, buffers_after)
        }; // queue lock dropped before user callback

        let recovered = callback_underrun_frames == 0 && stats.consecutive_underrun_callbacks > 0;
        if callback_underrun_frames > 0 {
            stats.underrun_frames += callback_underrun_frames;
            stats.underrun_callbacks += 1;
            stats.consecutive_underrun_callbacks += 1;

            // Per-generation totals reset on stream changes, so
            // every stream logs its first few underruns at debug.
            // That is intentional: startup underruns after a
            // clear/track change are the main diagnostic.
            if debug_logging
                && (should_log_sample(stats.underrun_callbacks)
                    || should_log_sample(stats.consecutive_underrun_callbacks))
            {
                log::debug!(
                                "Audio underrun: callback={}, missing_frames={} ({:.1}ms), queued_before={:.1}ms, queued_after={:.1}ms, buffers_after={}, cursor={:?}µs, generation={}, underrun_frames={}, underrun_callbacks={}, consecutive_underrun_callbacks={}",
                                stats.callbacks,
                                callback_underrun_frames,
                                callback_underrun_frames as f64 * 1000.0 / sample_rate as f64,
                                us_to_ms(queued_us),
                                us_to_ms(queued_after_us),
                                buffers_after,
                                cursor_us,
                                generation,
                                stats.underrun_frames,
                                stats.underrun_callbacks,
                                stats.consecutive_underrun_callbacks,
                            );
            }
        } else if recovered {
            let underrun_run = stats.consecutive_underrun_callbacks;
            stats.consecutive_underrun_callbacks = 0;
            log::debug!(
                            "Audio underrun recovered: callback={}, consecutive_underrun_callbacks={}, underrun_frames={}, queued_after={:.1}ms, buffers_after={}, generation={}",
                            stats.callbacks,
                            underrun_run,
                            stats.underrun_frames,
                            us_to_ms(queued_after_us),
                            buffers_after,
                            generation,
                        );
        }

        // Edge-triggered low-queue warnings with hysteresis, so a
        // queue hovering at one boundary cannot flood the log.
        if debug_logging {
            if !stats.queue_low && queued_after_us < QUEUE_LOW_WATER_US {
                stats.queue_low = true;
                log::debug!(
                                "Playback queue low: queued={:.1}ms, buffers={}, callback={}, underrun_frames={}, generation={}",
                                us_to_ms(queued_after_us),
                                buffers_after,
                                stats.callbacks,
                                stats.underrun_frames,
                                generation,
                            );
            } else if stats.queue_low && queued_after_us >= QUEUE_RECOVERED_WATER_US {
                stats.queue_low = false;
                log::debug!(
                                "Playback queue recovered: queued={:.1}ms, buffers={}, callback={}, generation={}",
                                us_to_ms(queued_after_us),
                                buffers_after,
                                stats.callbacks,
                                generation,
                            );
            }
        }

        // Apply gain with per-frame ramping
        // Apply gain with per-frame ramping
        let target = gain_control.gain();
        gain_ramp.apply(data, channels, target);

        if let Some(cb) = process_callback.as_mut() {
            cb(data);
        }

        // One sampled health line per rendered callback, emitted
        // after gain and the user process callback so it describes
        // the audio actually delivered.
        if trace_logging
            && callback_underrun_frames == 0
            && !recovered
            && should_log_sample(stats.callbacks)
        {
            let peak_abs = data.iter().map(|sample| sample.abs()).fold(0.0, f32::max);
            log::trace!(
                            "Audio callback rendered: callback={}, frames={}, queued_before={:.1}ms, queued_after={:.1}ms, buffers_after={}, peak_abs={:.6}, generation={}",
                            stats.callbacks,
                            frames,
                            us_to_ms(queued_us),
                            us_to_ms(queued_after_us),
                            buffers_after,
                            peak_abs,
                            generation,
                        );
        }
    }
}
