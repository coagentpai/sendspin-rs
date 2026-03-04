// ABOUTME: Backend-agnostic audio renderer with drift correction
// ABOUTME: Shared by cpal and PipeWire backends for synced playback

use crate::audio::gain::{GainControl, GainRamp};
use crate::audio::sync_correction::{CorrectionPlanner, CorrectionSchedule};
use crate::audio::{AudioBuffer, AudioFormat, Sample};
use crate::sync::ClockSync;
use parking_lot::Mutex;
use std::collections::VecDeque;
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
    queue: VecDeque<AudioBuffer>,
    current: Option<AudioBuffer>,
    index: usize,
    /// Current playback position in **server-time microseconds**. Periodically
    /// reanchored to the server's clock during clock-sync correction, so this
    /// represents "what server timestamp is playing right now", not how much
    /// audio content has been consumed.
    pub(crate) cursor_us: i64,
    pub(crate) cursor_remainder: i64,
    pub(crate) initialized: bool,
    pub(crate) generation: u64,
    pub(crate) force_reanchor: bool,
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
    }

    pub(crate) fn push(&mut self, buffer: AudioBuffer) {
        // Initialize the cursor from the first enqueued buffer so the audio
        // callback can see a valid cursor_us before it starts reading.
        if !self.initialized {
            self.cursor_us = buffer.timestamp;
            self.cursor_remainder = 0;
            self.initialized = true;
        }

        // When the server rebases its timeline backward, new chunks arrive with
        // timestamps that overlap chunks already in the queue. Remove all
        // overlapping buffers to prevent duplicate audio.
        let new_end = buffer.timestamp + buffer.duration_us();
        self.queue.retain(|b| {
            let existing_end = b.timestamp + b.duration_us();
            !(buffer.timestamp < existing_end && b.timestamp < new_end)
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

    pub(crate) fn next_frame(&mut self, channels: usize, sample_rate: u32) -> Option<&[Sample]> {
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

                // Skip past samples that are behind the cursor.
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

                // If the skip consumed the entire buffer, discard it and try next.
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
}

/// Backend-agnostic audio renderer with drift correction.
///
/// Called from both cpal and PipeWire process callbacks. The `playback_instant`
/// parameter abstracts the platform-specific timestamp:
/// - **cpal**: `Instant::now() + (playback - callback)` from `OutputCallbackInfo`
/// - **PipeWire**: `Instant::now()` (RT callbacks fire just-in-time)
pub(crate) struct AudioRenderer {
    queue: Arc<Mutex<PlaybackQueue>>,
    clock_sync: Arc<Mutex<ClockSync>>,
    channels: usize,
    sample_rate: u32,
    gain_control: GainControl,
    gain_ramp: GainRamp,
    planner: CorrectionPlanner,
    schedule: CorrectionSchedule,
    insert_counter: u32,
    drop_counter: u32,
    started: bool,
    last_generation: u64,
    last_frame: Vec<Sample>,
    process_callback: Option<ProcessCallback>,
}

impl AudioRenderer {
    pub(crate) fn new(
        queue: Arc<Mutex<PlaybackQueue>>,
        clock_sync: Arc<Mutex<ClockSync>>,
        format: &AudioFormat,
        gain_control: GainControl,
        process_callback: Option<ProcessCallback>,
    ) -> Self {
        let initial_gain = gain_control.target_gain();
        Self {
            queue,
            clock_sync,
            channels: format.channels as usize,
            sample_rate: format.sample_rate,
            gain_control,
            gain_ramp: GainRamp::new(format.sample_rate, initial_gain),
            planner: CorrectionPlanner::new(),
            schedule: CorrectionSchedule::default(),
            insert_counter: 0,
            drop_counter: 0,
            started: false,
            last_generation: 0,
            last_frame: vec![Sample::ZERO; format.channels as usize],
            process_callback,
        }
    }

    /// Core render method — called by both cpal and PipeWire backends.
    /// `playback_instant` is when these samples will physically play.
    pub(crate) fn render(&mut self, data: &mut [f32], playback_instant: Instant) {
        let channels = self.channels;
        let sample_rate = self.sample_rate;

        // Read all queue state in a single lock to avoid TOCTOU.
        let (generation, cursor_us, force_reanchor) = {
            let queue = self.queue.lock();
            let cursor = if queue.initialized {
                Some(queue.cursor_us)
            } else {
                None
            };
            (queue.generation, cursor, queue.force_reanchor)
        };

        if generation != self.last_generation {
            self.last_generation = generation;
            self.started = false;
            self.schedule = CorrectionSchedule::default();
            self.insert_counter = 0;
            self.drop_counter = 0;
            for sample in self.last_frame.iter_mut() {
                *sample = Sample::ZERO;
            }
        }

        // try_lock: skip sync if contended rather than blocking the audio thread.
        if let (Some(cursor_us), Some(sync)) = (cursor_us, self.clock_sync.try_lock()) {
            if let Some(expected_instant) = sync.server_to_local_instant(cursor_us) {
                let early_window = Duration::from_millis(1);
                if !self.started && playback_instant + early_window < expected_instant {
                    for sample in data.iter_mut() {
                        *sample = 0.0;
                    }
                    let target = self.gain_control.target_gain();
                    let frames = data.len() / channels;
                    self.gain_ramp.advance(frames, target);
                    if let Some(ref mut cb) = self.process_callback {
                        cb(data);
                    }
                    return;
                }
                self.started = true;

                let error_us = if playback_instant >= expected_instant {
                    playback_instant
                        .duration_since(expected_instant)
                        .as_micros() as i64
                } else {
                    -(expected_instant
                        .duration_since(playback_instant)
                        .as_micros() as i64)
                };
                let new_schedule =
                    self.planner
                        .plan(error_us, sample_rate, self.schedule.is_correcting());
                if new_schedule != self.schedule {
                    if new_schedule.is_correcting() != self.schedule.is_correcting() {
                        if new_schedule.is_correcting() {
                            log::debug!(
                                "Sync correction engaged: \
                                 error={:.1}ms, insert_every={}, drop_every={}",
                                error_us as f64 / 1000.0,
                                new_schedule.insert_every_n_frames,
                                new_schedule.drop_every_n_frames,
                            );
                        } else {
                            log::debug!(
                                "Sync correction disengaged: \
                                 error={:.1}ms",
                                error_us as f64 / 1000.0,
                            );
                        }
                    }
                    self.schedule = new_schedule;
                    self.insert_counter = self.schedule.insert_every_n_frames;
                    self.drop_counter = self.schedule.drop_every_n_frames;
                }

                if self.schedule.reanchor || force_reanchor {
                    if let Some(client_micros) = sync.instant_to_client_micros(playback_instant) {
                        if let Some(server_time) = sync.client_to_server_micros(client_micros) {
                            let mut queue = self.queue.lock();
                            queue.cursor_us = server_time;
                            queue.cursor_remainder = 0;
                            queue.force_reanchor = false;
                        }
                    }
                    self.schedule = CorrectionSchedule::default();
                    self.insert_counter = 0;
                    self.drop_counter = 0;
                }
            }
        }

        // If playback hasn't started yet, output silence.
        if !self.started {
            for sample in data.iter_mut() {
                *sample = 0.0;
            }
            let target = self.gain_control.target_gain();
            let frames = data.len() / channels;
            self.gain_ramp.advance(frames, target);
            if let Some(ref mut cb) = self.process_callback {
                cb(data);
            }
            return;
        }

        {
            let mut queue = self.queue.lock();
            let frames = data.len() / channels;
            let mut out_index = 0;

            for _ in 0..frames {
                if self.schedule.drop_every_n_frames > 0 {
                    self.drop_counter = self.drop_counter.saturating_sub(1);
                    if self.drop_counter == 0 {
                        // Discard one frame to catch up
                        let _ = queue.next_frame(channels, sample_rate);
                        self.drop_counter = self.schedule.drop_every_n_frames;
                        // Get and output the next frame
                        if let Some(frame) = queue.next_frame(channels, sample_rate) {
                            self.last_frame.copy_from_slice(frame);
                            for sample in frame {
                                data[out_index] = sample.to_f32();
                                out_index += 1;
                            }
                        } else {
                            for sample in &self.last_frame {
                                data[out_index] = sample.to_f32();
                                out_index += 1;
                            }
                        }
                        continue;
                    }
                }

                if self.schedule.insert_every_n_frames > 0 {
                    self.insert_counter = self.insert_counter.saturating_sub(1);
                    if self.insert_counter == 0 {
                        self.insert_counter = self.schedule.insert_every_n_frames;
                        for sample in &self.last_frame {
                            data[out_index] = sample.to_f32();
                            out_index += 1;
                        }
                        continue;
                    }
                }

                if let Some(frame) = queue.next_frame(channels, sample_rate) {
                    self.last_frame.copy_from_slice(frame);
                    for sample in frame {
                        data[out_index] = sample.to_f32();
                        out_index += 1;
                    }
                } else {
                    for _ in 0..channels {
                        data[out_index] = 0.0;
                        out_index += 1;
                    }
                }
            }
        } // queue lock dropped before user callback

        // Apply gain with per-frame ramping
        let target = self.gain_control.target_gain();
        self.gain_ramp.apply(data, channels, target);

        if let Some(ref mut cb) = self.process_callback {
            cb(data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PlaybackQueue;
    use crate::audio::{AudioBuffer, AudioFormat, Codec, Sample};
    use std::sync::Arc;
    use std::time::Instant;

    /// Standard test format: 48kHz stereo 24-bit PCM.
    fn test_format() -> AudioFormat {
        AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        }
    }

    /// Mono variant of [`test_format`].
    fn test_format_mono() -> AudioFormat {
        AudioFormat {
            channels: 1,
            ..test_format()
        }
    }

    #[test]
    fn test_queue_clear_bumps_generation() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![Sample::ZERO; 96];
        queue.push(AudioBuffer {
            timestamp: 1234,
            play_at: Instant::now(),
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        let before = queue.generation;
        queue.clear();
        assert_ne!(queue.generation, before);
        assert!(queue.queue.is_empty());
        assert!(!queue.initialized);
    }

    #[test]
    fn test_queue_drops_stale_buffers() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        let stale_samples: Vec<Sample> = (0..4800 * 2).map(|_| Sample(111)).collect();
        let fresh_samples: Vec<Sample> = (0..4800 * 2).map(|_| Sample(222)).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(stale_samples.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 200_000,
            play_at: Instant::now(),
            samples: Arc::from(fresh_samples.into_boxed_slice()),
            format,
        });

        queue.cursor_us = 150_000;
        queue.initialized = true;

        let frame_data: Vec<Sample> = queue
            .next_frame(2, 48_000)
            .expect("expected a frame")
            .to_vec();
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 200_000);
        assert_eq!(frame_data[0], Sample(222));
        assert_eq!(frame_data[1], Sample(222));
    }

    #[test]
    fn test_queue_push_sorts_by_timestamp() {
        let mut queue = PlaybackQueue::new();
        let format = test_format_mono();

        for ts in [300_000i64, 100_000, 200_000] {
            let samples = vec![Sample(ts as i32); 48];
            queue.push(AudioBuffer {
                timestamp: ts,
                play_at: Instant::now(),
                samples: Arc::from(samples.into_boxed_slice()),
                format: format.clone(),
            });
        }

        queue.cursor_us = 0;

        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 100_000);

        for _ in 1..48 {
            queue.next_frame(1, 48_000);
        }
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 200_000);

        for _ in 1..48 {
            queue.next_frame(1, 48_000);
        }
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 300_000);
    }

    #[test]
    fn test_queue_cursor_advances_correctly() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        let num_frames = 480;
        let samples = vec![Sample::ZERO; num_frames * 2];
        let start_ts = 1_000_000i64;
        queue.push(AudioBuffer {
            timestamp: start_ts,
            play_at: Instant::now(),
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        for _ in 0..num_frames {
            let _ = queue.next_frame(2, 48_000);
        }

        let expected_end = start_ts + 10_000;
        assert_eq!(
            queue.cursor_us,
            expected_end,
            "cursor should advance by exactly 10ms (10000us), got delta={}",
            queue.cursor_us - start_ts
        );
    }

    #[test]
    fn test_cursor_does_not_advance_during_underrun() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        let samples = vec![Sample::ZERO; 480 * 2];
        let start_ts = 1_000_000i64;
        queue.push(AudioBuffer {
            timestamp: start_ts,
            play_at: Instant::now(),
            samples: Arc::from(samples.into_boxed_slice()),
            format: format.clone(),
        });

        for _ in 0..480 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }
        let cursor_after_drain = queue.cursor_us;

        for _ in 0..1000 {
            assert!(queue.next_frame(2, 48_000).is_none());
        }
        assert_eq!(
            queue.cursor_us,
            cursor_after_drain,
            "cursor must not advance during underrun; advanced by {}us",
            queue.cursor_us - cursor_after_drain
        );

        let fresh_samples: Vec<Sample> = (0..480 * 2).map(|_| Sample(999)).collect();
        queue.push(AudioBuffer {
            timestamp: cursor_after_drain,
            play_at: Instant::now(),
            samples: Arc::from(fresh_samples.into_boxed_slice()),
            format,
        });

        let frame = queue
            .next_frame(2, 48_000)
            .expect("buffer should not be dropped as stale");
        assert_eq!(
            frame[0],
            Sample(999),
            "should get the fresh buffer, not stale data"
        );
    }

    #[test]
    fn test_push_initializes_cursor_from_first_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        assert!(!queue.initialized);
        assert_eq!(queue.cursor_us, 0);

        let samples = vec![Sample::ZERO; 96];
        queue.push(AudioBuffer {
            timestamp: 500_000,
            play_at: Instant::now(),
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        assert!(queue.initialized);
        assert_eq!(queue.cursor_us, 500_000);
        assert_eq!(queue.cursor_remainder, 0);
    }

    #[test]
    fn test_push_does_not_regress_cursor_after_init() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![Sample::ZERO; 96];

        queue.push(AudioBuffer {
            timestamp: 500_000,
            play_at: Instant::now(),
            samples: Arc::from(samples.clone().into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.cursor_us, 500_000);

        let _ = queue.next_frame(2, 48_000);
        let cursor_after_consume = queue.cursor_us;
        assert!(cursor_after_consume > 500_000);

        queue.push(AudioBuffer {
            timestamp: 200_000,
            play_at: Instant::now(),
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });
        assert_eq!(
            queue.cursor_us, cursor_after_consume,
            "cursor must not regress after playback has started"
        );
    }

    #[test]
    fn test_next_frame_skips_into_overlapping_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        let buf_a: Vec<Sample> = (0..2400 * 2).map(|_| Sample(111)).collect();
        let buf_b: Vec<Sample> = (0..2400 * 2)
            .map(|i| {
                if i < 2400 {
                    Sample(222)
                } else {
                    Sample(333)
                }
            })
            .collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(buf_a.into_boxed_slice()),
            format: format.clone(),
        });

        for _ in 0..2400 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }
        assert_eq!(queue.cursor_us, 50_000);

        queue.push(AudioBuffer {
            timestamp: 25_000,
            play_at: Instant::now(),
            samples: Arc::from(buf_b.into_boxed_slice()),
            format,
        });

        let frame = queue
            .next_frame(2, 48_000)
            .expect("should get a frame from buffer B");
        assert_eq!(
            frame[0],
            Sample(333),
            "expected skip into second half of buffer B (past the overlap), \
             got first half — backward-timestamped audio was replayed"
        );
    }

    #[test]
    fn test_next_frame_no_skip_when_buffer_starts_at_or_after_cursor() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        let samples_a: Vec<Sample> = (0..2400 * 2).map(|_| Sample(111)).collect();
        let samples_b: Vec<Sample> = (0..2400 * 2).map(|_| Sample(222)).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 50_000,
            play_at: Instant::now(),
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        for _ in 0..2400 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }

        let frame = queue
            .next_frame(2, 48_000)
            .expect("should get first frame of buffer B");
        assert_eq!(
            frame[0],
            Sample(222),
            "buffer B should play from the start (no skip needed)"
        );
    }

    #[test]
    fn test_push_dedup_replaces_overlapping_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        let samples_a: Vec<Sample> = (0..480 * 2).map(|_| Sample(111)).collect();
        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.queue.len(), 1);

        let samples_b: Vec<Sample> = (0..480 * 2).map(|_| Sample(222)).collect();
        queue.push(AudioBuffer {
            timestamp: 5_000,
            play_at: Instant::now(),
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].samples[0], Sample(222));
    }

    #[test]
    fn test_push_dedup_no_false_positive_small_chunks() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        let samples_a: Vec<Sample> = (0..240 * 2).map(|_| Sample(111)).collect();
        let samples_b: Vec<Sample> = (0..240 * 2).map(|_| Sample(222)).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 5_000,
            play_at: Instant::now(),
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.queue.len(), 2);
        assert_eq!(queue.queue[0].timestamp, 0);
        assert_eq!(queue.queue[1].timestamp, 5_000);
    }

    #[test]
    fn test_push_dedup_removes_all_overlapping() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        let samples_a: Vec<Sample> = (0..480 * 2).map(|_| Sample(111)).collect();
        let samples_b: Vec<Sample> = (0..480 * 2).map(|_| Sample(222)).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 12_000,
            play_at: Instant::now(),
            samples: Arc::from(samples_b.into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.queue.len(), 2);

        let samples_c: Vec<Sample> = (0..960 * 2).map(|_| Sample(333)).collect();
        queue.push(AudioBuffer {
            timestamp: 9_000,
            play_at: Instant::now(),
            samples: Arc::from(samples_c.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].timestamp, 9_000);
        assert_eq!(queue.queue[0].samples[0], Sample(333));
    }

    #[test]
    fn test_skip_past_entire_buffer_does_not_panic() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        let short_samples: Vec<Sample> = (0..48 * 2).map(|_| Sample(111)).collect();
        let ahead_samples: Vec<Sample> = (0..480 * 2).map(|_| Sample(222)).collect();

        queue.initialized = true;
        queue.cursor_us = 50_000;
        queue.queue.push_back(AudioBuffer {
            timestamp: 49_000,
            play_at: Instant::now(),
            samples: Arc::from(short_samples.into_boxed_slice()),
            format: format.clone(),
        });
        queue.queue.push_back(AudioBuffer {
            timestamp: 50_000,
            play_at: Instant::now(),
            samples: Arc::from(ahead_samples.into_boxed_slice()),
            format,
        });

        let frame = queue
            .next_frame(2, 48_000)
            .expect("should return a frame from the next buffer, not panic");
        assert_eq!(
            frame[0],
            Sample(222),
            "expected frame from the ahead buffer"
        );
    }
}
