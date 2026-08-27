// ABOUTME: Synced audio player with drift correction
// ABOUTME: Uses DAC callback timestamps to drop/insert frames for alignment

use crate::audio::gain::GainControl;
use crate::audio::renderer::{us_to_ms, AudioRenderer, PlaybackQueue};
use crate::audio::{AudioBuffer, AudioFormat};
use crate::error::Error;
use crate::log_sampling::should_log_sample;
use crate::sync::ClockSync;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use cpal::{Sample, I24};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub use crate::audio::renderer::ProcessCallback;

/// Maximum static delay in milliseconds. The Sendspin protocol defines
/// `static_delay_ms` over 0–5000; larger values are clamped to this.
pub const MAX_STATIC_DELAY_MS: u16 = 5000;

/// Configuration for [`SyncedPlayer`] construction.
pub struct SyncedPlayerConfig {
    /// Audio output device. Uses the platform default output device when `None`.
    pub device: Option<Device>,
    /// Initial playback volume, 0-100.
    pub volume: u8,
    /// Initial mute state.
    pub muted: bool,
    /// Optional fixed cpal buffer size in frames. When `None`, Windows
    /// requests a 40ms endpoint buffer (see `WINDOWS_DEFAULT_BUFFER_MS` for
    /// the rationale) and other platforms use the cpal device default.
    pub buffer_size: Option<u32>,
}

impl SyncedPlayerConfig {
    /// Create a config with common playback defaults.
    pub fn new() -> Self {
        Self {
            device: None,
            volume: 100,
            muted: false,
            buffer_size: None,
        }
    }
}

impl Default for SyncedPlayerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a protocol `static_delay_ms` value to microseconds, clamping to
/// [`MAX_STATIC_DELAY_MS`]. Out-of-range values are clamped rather than
/// rejected so a sloppy server can't disable playback timing entirely.
const fn static_delay_ms_to_us(delay_ms: u16) -> u64 {
    let clamped = if delay_ms > MAX_STATIC_DELAY_MS {
        MAX_STATIC_DELAY_MS
    } else {
        delay_ms
    };
    clamped as u64 * 1_000
}

/// Endpoint buffer requested on Windows when the caller does not override
/// [`SyncedPlayerConfig::buffer_size`].
///
/// WASAPI's shared-mode default allocates two engine periods (20ms at the
/// common 10ms period), so one late wake of the event-loop thread drains the
/// buffer completely and the next slip starves the mixer (silence insertion —
/// a real timeline displacement). Requesting ~40ms deepens the endpoint
/// queue only — the engine still wakes us every period — leaving roughly
/// three periods queued at each wake, so the mixer reliably survives two
/// consecutive late wakes (a third lands exactly at empty). That margin
/// matters even with MMCSS: the thread
/// shares the real-time class with every other pro-audio client and will
/// sometimes lose its slot.
///
/// Depth does not affect inter-device sync: the sync error is measured
/// against padding-compensated presentation timestamps, so queued depth
/// cancels out of the alignment math. The cost is bounded reaction latency —
/// a track-skip flush, volume ramp, or drift correction reaches the speaker
/// only after the already-queued frames drain.
const WINDOWS_DEFAULT_BUFFER_MS: u32 = 40;

/// Frames for [`WINDOWS_DEFAULT_BUFFER_MS`] at `sample_rate`.
const fn windows_default_buffer_frames(sample_rate: u32) -> u32 {
    sample_rate * WINDOWS_DEFAULT_BUFFER_MS / 1000
}

/// Endpoint buffer request when the caller does not specify one.
fn default_buffer_size(sample_rate: u32) -> cpal::BufferSize {
    if cfg!(target_os = "windows") {
        cpal::BufferSize::Fixed(windows_default_buffer_frames(sample_rate))
    } else {
        cpal::BufferSize::Default
    }
}

struct CallbackOutputs {
    error: Arc<Mutex<Option<String>>>,
}

/// Synced audio output with drift correction.
pub struct SyncedPlayer {
    format: AudioFormat,
    _stream: Stream,
    queue: Arc<Mutex<PlaybackQueue>>,
    /// Last error from the audio stream callback, if any.
    last_error: Arc<Mutex<Option<String>>>,
    gain: GainControl,
    /// Shared with the audio callback. See [`SyncedPlayer::set_static_delay`].
    static_delay_us: Arc<AtomicU64>,
}

impl SyncedPlayer {
    /// Create a new synced player using the provided clock sync and optional device.
    ///
    /// The player starts at `volume` (0-100) and `muted` state. These are
    /// applied immediately — the first audio callback uses the correct gain
    /// with no ramp from a default value.
    /// The `buffer_size` overrides the endpoint buffer request. If not set,
    /// Windows requests 40ms of margin (see `WINDOWS_DEFAULT_BUFFER_MS` for
    /// the rationale) and other platforms use the cpal device default.
    pub fn new(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedPlayerConfig,
    ) -> Result<Self, Error> {
        Self::build(format, clock_sync, config, None)
    }

    /// Create a player with a process callback for post-gain audio processing.
    ///
    /// The callback receives samples **after** gain/mute processing has been
    /// applied. See [`ProcessCallback`] for thread-safety requirements.
    ///
    /// # Example (requires physical audio hardware to run)
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use parking_lot::Mutex;
    /// # use sendspin::audio::{AudioFormat, Codec, SyncedPlayer, SyncedPlayerConfig};
    /// # use sendspin::sync::ClockSync;
    /// # use sendspin::DefaultClock;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let format = AudioFormat {
    ///     codec: Codec::Pcm,
    ///     sample_rate: 48_000,
    ///     channels: 2,
    ///     bit_depth: 24,
    ///     codec_header: None,
    /// };
    /// let clock_sync = Arc::new(Mutex::new(ClockSync::new(Arc::new(DefaultClock::new()))));
    /// let player = SyncedPlayer::with_process_callback(
    ///     format,
    ///     clock_sync,
    ///     SyncedPlayerConfig::new(),
    ///     Box::new(|data| { /* e.g. feed a VU meter or visualizer */ }),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_process_callback(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedPlayerConfig,
        callback: ProcessCallback,
    ) -> Result<Self, Error> {
        Self::build(format, clock_sync, config, Some(callback))
    }

    fn build(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedPlayerConfig,
        process_callback: Option<ProcessCallback>,
    ) -> Result<Self, Error> {
        if format.channels == 0 {
            return Err(Error::Output("channels must be > 0".to_string()));
        }
        let host = cpal::default_host();
        let device = match config.device {
            Some(device) => device,
            None => host
                .default_output_device()
                .ok_or_else(|| Error::Output("No output device available".to_string()))?,
        };

        let stream_config = StreamConfig {
            channels: format.channels as u16,
            sample_rate: cpal::SampleRate::from(format.sample_rate),
            buffer_size: match config.buffer_size {
                Some(frames) => cpal::BufferSize::Fixed(frames),
                None => default_buffer_size(format.sample_rate),
            },
        };

        let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
        let format_clone = format.clone();
        let last_error = Arc::new(Mutex::new(None));
        let gain = GainControl::new(config.volume, config.muted);
        let static_delay_us = Arc::new(AtomicU64::new(0));

        let renderer = AudioRenderer::new(
            Arc::clone(&queue),
            clock_sync,
            &format,
            gain.clone(),
            process_callback,
            Arc::clone(&static_delay_us),
        );
        let callback_outputs = CallbackOutputs {
            error: Arc::clone(&last_error),
        };

        let stream = Self::build_stream(
            &device,
            &stream_config,
            renderer,
            format_clone,
            callback_outputs,
        )?;
        stream.play().map_err(|e| Error::Output(e.to_string()))?;
        log::info!(
            "SyncedPlayer started: {} channels, {} Hz, {}-bit",
            format.channels,
            format.sample_rate,
            format.bit_depth,
        );

        Ok(Self {
            format,
            _stream: stream,
            queue,
            last_error,
            gain,
            static_delay_us,
        })
    }

    /// Enqueue a decoded buffer for playback.
    ///
    /// Scheduling uses `buffer.timestamp` (server time in microseconds) for
    /// drift-corrected playback.
    pub fn enqueue(&self, buffer: AudioBuffer) {
        let buffer_timestamp = buffer.timestamp;
        let buffer_duration_us = buffer.duration_us();
        let channels = self.format.channels as usize;
        let sample_rate = self.format.sample_rate;

        // Snapshot log fields under the lock but log after dropping it: the
        // audio callback contends on this lock, and logging can block on I/O.
        // The O(buffers) depth walk runs only for sampled, trace-enabled
        // enqueues.
        let trace_fields = {
            let mut queue = self.queue.lock();
            queue.push(buffer);
            queue.enqueue_count += 1;
            if log::log_enabled!(log::Level::Trace) && should_log_sample(queue.enqueue_count) {
                Some((
                    queue.enqueue_count,
                    queue.queued_duration_us(channels, sample_rate),
                    queue.buffer_count(),
                    queue.cursor_us,
                    queue.generation,
                ))
            } else {
                None
            }
        };

        if let Some((enqueue_count, queued_us, buffers, cursor_us, generation)) = trace_fields {
            log::trace!(
                "Audio buffer enqueued: enqueue={}, ts={}µs, duration={:.1}ms, queued={:.1}ms, buffers={}, cursor={}µs, generation={}",
                enqueue_count,
                buffer_timestamp,
                buffer_duration_us as f64 / 1000.0,
                us_to_ms(queued_us),
                buffers,
                cursor_us,
                generation,
            );
        }
    }

    /// Clear queued audio and reset playback state.
    pub fn clear(&self) {
        let channels = self.format.channels as usize;
        let sample_rate = self.format.sample_rate;

        // Snapshot log fields under the lock but log after dropping it: the
        // audio callback contends on this lock, and logging can block on I/O.
        let debug_fields = {
            let mut queue = self.queue.lock();
            let fields = log::log_enabled!(log::Level::Debug).then(|| {
                (
                    queue.queued_duration_us(channels, sample_rate),
                    queue.buffer_count(),
                    queue.cursor_us,
                    queue.generation,
                )
            });
            queue.clear();
            fields
        };

        if let Some((queued_us, buffers, cursor_us, generation)) = debug_fields {
            log::debug!(
                "Cleared playback queue: queued={:.1}ms, buffers={}, cursor={}µs, generation={}",
                us_to_ms(queued_us),
                buffers,
                cursor_us,
                generation,
            );
        }
    }

    /// Release the underlying audio output device.
    ///
    /// Consuming the player drops its active `cpal::Stream`, allowing another
    /// local source to open the same device. Use this before advertising
    /// `state: "external_source"`.
    ///
    /// The returned [`GainControl`] preserves Sendspin's software volume/mute
    /// state across the handoff. It does not reflect hardware or OS mixer
    /// changes made by the external source.
    pub fn release_audio_device(self) -> GainControl {
        self.gain
    }

    /// Return the configured audio format.
    pub fn format(&self) -> &AudioFormat {
        &self.format
    }

    /// Check if the audio stream has encountered an error.
    ///
    /// Returns the error message if one occurred, clearing it in the process.
    pub fn take_error(&self) -> Option<String> {
        self.last_error.lock().take()
    }

    /// Check if the audio stream has an error without clearing it.
    pub fn has_error(&self) -> bool {
        self.last_error.lock().is_some()
    }

    /// Get a reference to the volume/mute control.
    ///
    /// Call `.clone()` if you need an owned handle to share across threads
    /// (cloning is cheap — single `Arc` increment, no data copy).
    pub fn gain_control(&self) -> &GainControl {
        &self.gain
    }

    // -- Volume/mute convenience methods --
    //
    // These promote the most common operations for ergonomics in simple
    // use-cases. For full control, use `gain_control()` directly.

    /// Current volume as 0-100.
    pub fn volume(&self) -> u8 {
        self.gain.volume()
    }

    /// Whether playback is currently muted.
    pub fn is_muted(&self) -> bool {
        self.gain.is_muted()
    }

    /// Set playback volume (0-100).
    pub fn set_volume(&self, volume: u8) {
        self.gain.set_volume(volume);
    }

    /// Set mute state.
    pub fn set_mute(&self, muted: bool) {
        self.gain.set_mute(muted);
    }

    /// Set the static playback delay in milliseconds (0–[`MAX_STATIC_DELAY_MS`]).
    ///
    /// Compensates for external speaker/amplifier latency: the server pre-sends
    /// audio by this amount, so the player shifts each sample's emission earlier
    /// by the same delay to keep alignment correct. Values above the maximum are
    /// clamped. Takes effect on the next audio callback.
    ///
    /// A delay change is an intentional local timing offset, not clock drift.
    /// Request a one-shot reanchor so the audio callback either skips forward or
    /// waits for the new target time instead of feeding the delay delta through
    /// pitch-shifting drift correction.
    pub fn set_static_delay(&self, delay_ms: u16) {
        self.static_delay_us
            .store(static_delay_ms_to_us(delay_ms), Ordering::Relaxed);

        let mut queue = self.queue.lock();
        if queue.initialized {
            queue.force_reanchor = true;
        }
    }

    /// Current static delay in milliseconds.
    pub fn static_delay_ms(&self) -> u16 {
        (self.static_delay_us.load(Ordering::Relaxed) / 1_000) as u16
    }

    fn build_stream(
        device: &Device,
        config: &StreamConfig,
        mut renderer: AudioRenderer,
        format: AudioFormat,
        outputs: CallbackOutputs,
    ) -> Result<Stream, Error> {
        let CallbackOutputs { error } = outputs;
        // Staging buffer: the renderer works in f32; convert to the device's
        // sample format after rendering.
        let mut f32_buffer = Vec::<f32>::new();
        let device_config = device
            .default_output_config()
            .map_err(|e| Error::Output(e.to_string()))?;
        let mut stream_config = device_config.config();
        stream_config.buffer_size = config.buffer_size;
        stream_config.channels = format.channels.into();
        stream_config.sample_rate = format.sample_rate;

        macro_rules! output_stream {
            ($sample:ty) => {
                device
                    .build_output_stream(
                        stream_config,
                        move |data: &mut [$sample], info: &cpal::OutputCallbackInfo| {
                            let callback_instant = Instant::now();
                            let ts = info.timestamp();
                            let playback_delta = ts.playback.duration_since(ts.callback);
                            f32_buffer.clear();
                            f32_buffer.resize(data.len(), 0.0);
                            renderer.render(&mut f32_buffer, callback_instant, playback_delta);
                            for (dst, &sample) in data.iter_mut().zip(f32_buffer.iter()) {
                                *dst = <$sample>::from_sample(sample);
                            }
                        },
                        move |err| {
                            // cpal reports a refused real-time promotion as
                            // RealtimeDenied ("Audio will still play"); playback
                            // continues at normal priority. Warn without storing:
                            // take_error()/has_error() signal fatal stream
                            // failures, and a consumer must not tear down a
                            // working stream over a scheduling downgrade.
                            if err.kind() == cpal::ErrorKind::RealtimeDenied {
                                log::warn!(
                                    "Audio thread priority promotion failed (non-fatal): {err}"
                                );
                                return;
                            }
                            log::error!("Audio stream error: {err}");
                            *error.lock() = Some(err.to_string());
                        },
                        None,
                    )
                    .map_err(|e| Error::Output(e.to_string()))
            };
        }

        log::debug!(
            "Using output device: {}, config: {:?}",
            device
                .id()
                .map(|id| format!("{:?}", id))
                .map_err(|e| Error::Output(e.to_string()))?,
            device_config,
        );
        match device_config.sample_format() {
            SampleFormat::F32 => output_stream!(f32),
            SampleFormat::F64 => output_stream!(f64),
            SampleFormat::I8 => output_stream!(i8),
            SampleFormat::I16 => output_stream!(i16),
            SampleFormat::I24 => output_stream!(I24),
            SampleFormat::I32 => output_stream!(i32),
            SampleFormat::I64 => output_stream!(i64),
            SampleFormat::U8 => output_stream!(u8),
            SampleFormat::U16 => output_stream!(u16),
            SampleFormat::U32 => output_stream!(u32),
            SampleFormat::U64 => output_stream!(u64),
            _ => Err(Error::Output(format!(
                "Unsupported sample format: {:?}",
                device_config.sample_format()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    // Note: SyncedPlayer's convenience methods (volume, is_muted, set_volume,
    // set_mute, gain_control) delegate to GainControl which is thoroughly tested
    // in gain.rs. Wiring tests require a real audio device (cpal Stream) and
    // cannot run in CI.

    use super::{
        static_delay_ms_to_us, windows_default_buffer_frames, PlaybackQueue, SyncedPlayer,
        SyncedPlayerConfig, MAX_STATIC_DELAY_MS,
    };
    use crate::audio::{AudioBuffer, AudioFormat, Codec};
    use crate::error::Error;
    use crate::sync::{ClockSync, DefaultClock};
    use cpal::Sample;
    use parking_lot::Mutex;
    use std::sync::Arc;

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
    fn test_build_rejects_zero_channels() {
        // `SyncedPlayer::build` short-circuits on channels==0 *before* any
        // cpal device access, so this test works regardless of whether the
        // runner has audio hardware. Asserting on the specific error message
        // (not just `is_err()`) pins the check to the explicit guard — any
        // later failure (missing device, cpal rejecting the config) would
        // surface a different message.
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 0,
            bit_depth: 24,
            codec_header: None,
        };
        let clock_sync = Arc::new(Mutex::new(ClockSync::new(Arc::new(DefaultClock::new()))));
        let result = SyncedPlayer::new(format, clock_sync, SyncedPlayerConfig::new());
        let err = match result {
            Ok(_) => panic!("channels=0 should be rejected"),
            Err(e) => e,
        };
        match err {
            Error::Output(msg) => assert!(
                msg.contains("channels must be > 0"),
                "expected 'channels must be > 0' error, got: {msg}"
            ),
            other => panic!("expected Error::Output, got {other:?}"),
        }
    }

    #[test]
    fn test_windows_default_buffer_frames_is_40ms() {
        assert_eq!(windows_default_buffer_frames(44_100), 1_764);
        assert_eq!(windows_default_buffer_frames(48_000), 1_920);
        assert_eq!(windows_default_buffer_frames(192_000), 7_680);
    }

    #[test]
    fn test_static_delay_ms_to_us_converts_and_clamps() {
        // Nominal values convert milliseconds to microseconds.
        assert_eq!(static_delay_ms_to_us(0), 0);
        assert_eq!(static_delay_ms_to_us(100), 100_000);
        assert_eq!(
            static_delay_ms_to_us(MAX_STATIC_DELAY_MS),
            MAX_STATIC_DELAY_MS as u64 * 1_000
        );

        // Out-of-range values clamp to the maximum rather than overflow or wrap.
        assert_eq!(
            static_delay_ms_to_us(MAX_STATIC_DELAY_MS + 1),
            MAX_STATIC_DELAY_MS as u64 * 1_000
        );
        assert_eq!(
            static_delay_ms_to_us(u16::MAX),
            MAX_STATIC_DELAY_MS as u64 * 1_000
        );
    }

    #[test]
    fn test_queue_clear_bumps_generation() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 96];
        queue.push(AudioBuffer {
            timestamp: 1234,
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

        // Use distinct sample values so we can verify which buffer was returned.
        // 4800 stereo frames at 48kHz = 100ms per buffer.
        // With cursor at 150ms, the first buffer (ts=0, ends at 100ms) is stale.
        let stale_samples: Vec<i32> = (0..4800 * 2).map(|_| 111).collect();
        let fresh_samples: Vec<i32> = (0..4800 * 2).map(|_| 222).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(stale_samples.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 200_000,
            samples: Arc::from(fresh_samples.into_boxed_slice()),
            format,
        });

        queue.cursor_us = 150_000;
        queue.initialized = true;

        // Copy the frame data so we can release the mutable borrow on queue.
        let frame_data: Vec<i32> = queue
            .next_frame(2, 48_000)
            .expect("expected a frame")
            .to_vec();
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 200_000);
        // Verify we got the fresh buffer's data, not the stale one
        assert_eq!(frame_data[0], 222);
        assert_eq!(frame_data[1], 222);
    }

    #[test]
    fn test_queue_push_sorts_by_timestamp() {
        let mut queue = PlaybackQueue::new();
        let format = test_format_mono();

        // Push out of order: 300, 100, 200
        for ts in [300_000i64, 100_000, 200_000] {
            let samples = vec![ts as i32; 48]; // 1ms of mono
            queue.push(AudioBuffer {
                timestamp: ts,
                samples: Arc::from(samples.into_boxed_slice()),
                format: format.clone(),
            });
        }

        // Reset cursor so stale-buffer-dropping doesn't interfere with
        // the sort-order verification (in real usage, sync reanchor sets
        // cursor before playback begins).
        queue.cursor_us = 0;

        // Drain and verify sorted order
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 100_000);

        // Exhaust the first buffer (48 frames)
        for _ in 1..48 {
            queue.next_frame(1, 48_000);
        }
        // Next frame should come from the second buffer
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 200_000);

        // Exhaust the second buffer
        for _ in 1..48 {
            queue.next_frame(1, 48_000);
        }
        // Next frame should come from the third buffer
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 300_000);
    }

    #[test]
    fn test_queue_cursor_advances_correctly() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // 480 stereo frames = 10ms at 48kHz
        let num_frames = 480;
        let samples = vec![i32::EQUILIBRIUM; num_frames * 2];
        let start_ts = 1_000_000i64; // 1 second
        queue.push(AudioBuffer {
            timestamp: start_ts,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        // Consume all frames
        for _ in 0..num_frames {
            let _ = queue.next_frame(2, 48_000);
        }

        // 480 frames at 48kHz = 10,000us = 10ms
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

        // Push one buffer to initialize the cursor
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo
        let start_ts = 1_000_000i64;
        queue.push(AudioBuffer {
            timestamp: start_ts,
            samples: Arc::from(samples.into_boxed_slice()),
            format: format.clone(),
        });

        // Consume all frames
        for _ in 0..480 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }
        let cursor_after_drain = queue.cursor_us;

        // Queue is now empty. Calling next_frame should return None
        // and NOT advance the cursor.
        for _ in 0..1000 {
            assert!(queue.next_frame(2, 48_000).is_none());
        }
        assert_eq!(
            queue.cursor_us,
            cursor_after_drain,
            "cursor must not advance during underrun; advanced by {}us",
            queue.cursor_us - cursor_after_drain
        );

        // Push a new buffer after the underrun. It should NOT be
        // dropped as stale — the cursor hasn't raced ahead.
        let fresh_samples: Vec<i32> = (0..480 * 2).map(|_| 999).collect();
        queue.push(AudioBuffer {
            timestamp: cursor_after_drain, // starts right where we left off
            samples: Arc::from(fresh_samples.into_boxed_slice()),
            format,
        });

        let frame = queue
            .next_frame(2, 48_000)
            .expect("buffer should not be dropped as stale");
        assert_eq!(frame[0], 999, "should get the fresh buffer, not stale data");
    }

    #[test]
    fn test_push_initializes_cursor_from_first_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        assert!(!queue.initialized);
        assert_eq!(queue.cursor_us, 0);

        let samples = vec![i32::EQUILIBRIUM; 96];
        queue.push(AudioBuffer {
            timestamp: 500_000,
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
        let samples = vec![i32::EQUILIBRIUM; 96];

        // First buffer at 500ms — initializes cursor
        queue.push(AudioBuffer {
            timestamp: 500_000,
            samples: Arc::from(samples.clone().into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.cursor_us, 500_000);

        // Consume a frame so cursor advances past init
        let _ = queue.next_frame(2, 48_000);
        let cursor_after_consume = queue.cursor_us;
        assert!(cursor_after_consume > 500_000);

        // Push an earlier buffer — cursor must NOT regress
        queue.push(AudioBuffer {
            timestamp: 200_000,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });
        assert_eq!(
            queue.cursor_us, cursor_after_consume,
            "cursor must not regress after playback has started"
        );
    }

    #[test]
    fn test_first_playable_cursor_skips_stale_audio() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.first_playable_cursor_at_or_after(5_000), Some(5_000));
        assert_eq!(queue.first_playable_cursor_at_or_after(10_000), None);
    }

    #[test]
    fn test_first_playable_cursor_waits_for_future_audio() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo

        queue.push(AudioBuffer {
            timestamp: 20_000,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.first_playable_cursor_at_or_after(5_000), Some(20_000));
    }

    #[test]
    fn test_first_playable_cursor_uses_current_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        // Pull one frame so the buffer moves from `queue` into `current`,
        // exercising the `current` branch of first_playable_cursor_at_or_after.
        let _ = queue.next_frame(2, 48_000);
        assert!(queue.current.is_some());
        assert!(queue.queue.is_empty());

        assert_eq!(queue.first_playable_cursor_at_or_after(5_000), Some(5_000));
        assert_eq!(queue.first_playable_cursor_at_or_after(10_000), None);
    }

    #[test]
    fn test_first_playable_cursor_does_not_rewind_current_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        for _ in 0..240 {
            let _ = queue.next_frame(2, 48_000);
        }
        assert_eq!(queue.cursor_us, 5_000);

        assert_eq!(queue.first_playable_cursor_at_or_after(1_000), Some(5_000));
        assert_eq!(queue.first_playable_cursor_at_or_after(6_000), Some(6_000));
        assert_eq!(queue.first_playable_cursor_at_or_after(10_000), None);
    }

    #[test]
    fn test_next_frame_skips_into_overlapping_buffer() {
        // Simulates a backward timestamp jump from a server timeline rebase.
        // Buffer A is 50ms (2400 frames stereo at 48kHz). After consuming A
        // the cursor is at 50ms. Buffer B arrives at 25ms — the skip logic
        // should jump 25ms (1200 frames) into B.
        //
        // B is pushed AFTER A is consumed so dedup doesn't apply (A is no
        // longer in the queue).
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        let buf_a: Vec<i32> = (0..2400 * 2).map(|_| 111).collect();
        let buf_b: Vec<i32> = (0..2400 * 2)
            .map(|i| {
                // First half (1200 frames) = 222, second half = 333
                if i < 2400 {
                    222
                } else {
                    333
                }
            })
            .collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(buf_a.into_boxed_slice()),
            format: format.clone(),
        });

        // Consume all of buffer A (2400 frames). Cursor advances to 50000µs.
        for _ in 0..2400 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }
        assert_eq!(queue.cursor_us, 50_000);

        // Push B after A is consumed — no dedup, tests skip logic only.
        // push() no longer regresses cursor_us after init, so cursor stays
        // at 50000 and the skip logic activates naturally.
        queue.push(AudioBuffer {
            timestamp: 25_000,
            samples: Arc::from(buf_b.into_boxed_slice()),
            format,
        });

        // Buffer B starts at 25ms but cursor is at 50ms, so 25ms (1200 frames)
        // should be skipped. First returned frame should be Sample(333).
        let frame = queue
            .next_frame(2, 48_000)
            .expect("should get a frame from buffer B");
        assert_eq!(
            frame[0], 333,
            "expected skip into second half of buffer B (past the overlap), \
             got first half — backward-timestamped audio was replayed"
        );
    }

    #[test]
    fn test_next_frame_no_skip_when_buffer_starts_at_or_after_cursor() {
        // Verify that the skip logic doesn't activate for normal (non-overlapping)
        // buffers — only for buffers that start before the cursor.
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // 2400 frames = 50ms per buffer. Adjacent, non-overlapping.
        let samples_a: Vec<i32> = (0..2400 * 2).map(|_| 111).collect();
        let samples_b: Vec<i32> = (0..2400 * 2).map(|_| 222).collect();

        // Two consecutive, non-overlapping buffers.
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 50_000, // starts exactly where A ends
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        // Consume all of buffer A.
        for _ in 0..2400 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }

        // Buffer B starts at cursor (50000µs) — no skip should occur.
        let frame = queue
            .next_frame(2, 48_000)
            .expect("should get first frame of buffer B");
        assert_eq!(
            frame[0], 222,
            "buffer B should play from the start (no skip needed)"
        );
    }

    #[test]
    fn test_push_dedup_replaces_overlapping_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Buffer A: 10ms at ts=0 (480 stereo frames)
        let samples_a: Vec<i32> = (0..480 * 2).map(|_| 111).collect();
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.queue.len(), 1);

        // Buffer B: 10ms at ts=5000 (5ms) — overlaps A's range [0, 10000)
        let samples_b: Vec<i32> = (0..480 * 2).map(|_| 222).collect();
        queue.push(AudioBuffer {
            timestamp: 5_000,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        // Should replace A, not add a second entry
        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].samples[0], 222);
    }

    #[test]
    fn test_push_dedup_no_false_positive_small_chunks() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Two adjacent 5ms chunks (240 stereo frames each).
        // Chunk A: [0, 5000), Chunk B: [5000, 10000) — no overlap.
        let samples_a: Vec<i32> = (0..240 * 2).map(|_| 111).collect();
        let samples_b: Vec<i32> = (0..240 * 2).map(|_| 222).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 5_000,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        // Both should be kept — they're adjacent, not overlapping
        assert_eq!(queue.queue.len(), 2);
        assert_eq!(queue.queue[0].timestamp, 0);
        assert_eq!(queue.queue[1].timestamp, 5_000);
    }

    #[test]
    fn test_push_keeps_chunks_on_44_1k_floor_timestamp_grid() {
        // Regression for the field "continuous popping" report: aiosendspin's
        // 25ms chunks at 44.1kHz are 1102 frames = 24988.66µs, and its
        // floor-based timestamp grid advances 24988µs for a third of chunks
        // while `duration_us` rounds every chunk to 24989µs. Those chunks
        // start 1µs "inside" their predecessor; the phantom overlap must not
        // evict anything (pre-fix it discarded ~34% of all queued audio).
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 44_100,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        };

        // Mirror the server's residue arithmetic:
        // delta, residue = divmod(residue + frames * 1e6, rate).
        let mut ts = 282_697_405_880_i64; // first chunk ts from the field trace
        let mut residue = 0_i64;
        for _ in 0..12 {
            let samples: Vec<i32> = vec![0; 1102 * 2];
            queue.push(AudioBuffer {
                timestamp: ts,
                samples: Arc::from(samples.into_boxed_slice()),
                format: format.clone(),
            });
            residue += 1102 * 1_000_000;
            ts += residue / 44_100;
            residue %= 44_100;
        }

        assert_eq!(
            queue.queue.len(),
            12,
            "sub-frame timestamp jitter must never evict queued audio"
        );
    }

    #[test]
    fn test_push_keeps_floor_grid_chunks_at_all_supported_rates() {
        // The 44.1k regression swept across the whole rate family. 25ms
        // chunks are floor(rate/40) frames; the server grid advances by
        // floor-based (divmod) deltas. The mismatch between that grid and
        // our round-to-nearest duration_us is at most 1µs, which must stay
        // below the one-frame eviction threshold at every rate. (Only 44.1k
        // and 22.05k actually exhibit the mismatch; the rest divide 25ms
        // evenly and are exact.)
        for &rate in &[
            8_000_u32, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000,
            176_400, 192_000,
        ] {
            let mut queue = PlaybackQueue::new();
            let format = AudioFormat {
                codec: Codec::Pcm,
                sample_rate: rate,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            };
            let frames = (rate / 40) as usize;
            let mut ts = 1_000_000_i64;
            let mut residue = 0_i64;
            for _ in 0..16 {
                queue.push(AudioBuffer {
                    timestamp: ts,
                    samples: Arc::from(vec![0_i32; frames * 2].into_boxed_slice()),
                    format: format.clone(),
                });
                residue += frames as i64 * 1_000_000;
                ts += residue / i64::from(rate);
                residue %= i64::from(rate);
            }
            assert_eq!(
                queue.queue.len(),
                16,
                "rate {rate}: floor-grid timestamp jitter must not evict chunks"
            );
        }
    }

    #[test]
    fn test_push_dedup_still_evicts_one_frame_overlap_at_44_1k() {
        // Counterpart to the floor-grid regression: an overlap of a full
        // frame or more is real duplicate audio and must still dedup.
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 44_100,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        };

        let samples_a: Vec<i32> = vec![111; 1102 * 2];
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        // Chunk A spans [0, 24989). A chunk rebased 1ms back into it
        // overlaps by ~1ms ≫ one frame (23µs): evict A.
        let samples_b: Vec<i32> = vec![222; 1102 * 2];
        queue.push(AudioBuffer {
            timestamp: 23_989,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].timestamp, 23_989);
    }

    #[test]
    fn test_push_dedup_removes_all_overlapping() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Buffer A: 10ms at ts=0 — range [0, 10000)
        let samples_a: Vec<i32> = (0..480 * 2).map(|_| 111).collect();
        // Buffer B: 10ms at ts=12000 — range [12000, 22000). No overlap with A.
        let samples_b: Vec<i32> = (0..480 * 2).map(|_| 222).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 12_000,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.queue.len(), 2);

        // Buffer C: 20ms at ts=9000 — range [9000, 29000).
        // Overlaps both A (9000 < 10000 && 0 < 29000) and B (9000 < 22000 && 12000 < 29000).
        // Both stale buffers should be removed — the server will send fresh
        // data for any gaps. Keeping either would cause duplicate audio.
        let samples_c: Vec<i32> = (0..960 * 2).map(|_| 333).collect();
        queue.push(AudioBuffer {
            timestamp: 9_000,
            samples: Arc::from(samples_c.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].timestamp, 9_000);
        assert_eq!(queue.queue[0].samples[0], 333);
    }

    #[test]
    fn test_push_keeps_adjacent_buffers_inserted_in_reverse_order() {
        // When buffer B is pushed *after* buffer A and they abut at the
        // boundary (B.ts == A.end), the existing dedup check keeps both
        // because `b.timestamp < new_end` is false at the boundary. The
        // symmetric case — pushing the *earlier* buffer second — must also
        // keep both: when we push the earlier buffer A and retain() walks
        // B, we see `B.ts == new_end` (A.end). The dedup has to treat the
        // boundary as *not* overlapping, otherwise we'd evict a buffer that
        // simply abuts — a common pattern when chunks arrive out-of-order.
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Two 5ms adjacent chunks: A at [0, 5000), B at [5000, 10000).
        let samples_a: Vec<i32> = (0..240 * 2).map(|_| 111).collect();
        let samples_b: Vec<i32> = (0..240 * 2).map(|_| 222).collect();

        // Push the *later* buffer (B) first …
        queue.push(AudioBuffer {
            timestamp: 5_000,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format: format.clone(),
        });
        // … then push the *earlier* buffer (A). A.end == B.ts (boundary case).
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format,
        });

        assert_eq!(
            queue.queue.len(),
            2,
            "adjacent buffers pushed in reverse order should both be kept"
        );
        // Should also be sorted: A first, then B.
        assert_eq!(queue.queue[0].timestamp, 0);
        assert_eq!(queue.queue[1].timestamp, 5_000);
    }

    #[test]
    fn test_next_frame_returns_final_frame_when_skip_lands_at_last_frame() {
        // When the cursor skip lands `self.index` at exactly
        // `samples.len() - channels`, there is still one playable frame at
        // the tail of the buffer. The "is this buffer exhausted?" check in
        // the outer match is `self.index + channels > c.samples.len()` —
        // strictly greater — so `index == samples.len() - channels` falls
        // through to the frame-return path. A non-strict comparison here
        // would silently drop the last frame of every buffer whose skip
        // landed on the final-frame boundary.
        //
        // Setup: 48-frame stereo buffer at ts=0, cursor at 980 µs.
        //   skip_us     = 980 - 0                          = 980
        //   skip_frames = 980 * 48_000 / 1_000_000         = 47
        //   self.index  = 47 * 2                           = 94
        //   samples.len = 48 * 2                           = 96
        //   94 + 2 == 96 — keep buffer, return last frame.
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Distinctive sample values so we can assert *which* frame was returned.
        let samples: Vec<i32> = (0..48 * 2).map(|_| 111).collect();

        queue.initialized = true;
        queue.cursor_us = 980;
        queue.queue.push_back(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        let frame = queue
            .next_frame(2, 48_000)
            .expect("last frame should be returned, not discarded");
        // Final stereo frame — indices 94 and 95 of the flat sample array.
        assert_eq!(frame, &[111, 111]);
    }

    #[test]
    fn test_skip_past_entire_buffer_does_not_panic() {
        // When the cursor is far ahead of a buffer, the skip logic can set
        // self.index past the buffer's sample count. next_frame must not
        // panic; it should discard the buffer and return the next one.
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // 1ms buffer (48 stereo frames) at ts=49000. Duration = 1000µs,
        // so it ends at 50000 which is NOT < cursor (50000), surviving
        // the stale-drop. But the skip logic sees ts=49000 < cursor=50000
        // and tries to skip 1ms (48 frames) — exactly the buffer length.
        let short_samples: Vec<i32> = (0..48 * 2).map(|_| 111).collect();
        // Buffer that starts at cursor: 10ms at ts=50000
        let ahead_samples: Vec<i32> = (0..480 * 2).map(|_| 222).collect();

        queue.initialized = true;
        queue.cursor_us = 50_000;
        queue.queue.push_back(AudioBuffer {
            timestamp: 49_000,
            samples: Arc::from(short_samples.into_boxed_slice()),
            format: format.clone(),
        });
        queue.queue.push_back(AudioBuffer {
            timestamp: 50_000,
            samples: Arc::from(ahead_samples.into_boxed_slice()),
            format,
        });

        // The skip tries to skip 1ms (48 frames) into a 48-frame buffer —
        // index lands at the end. Must not panic; should discard the short
        // buffer and return from the next one (ts=50000).
        let frame = queue
            .next_frame(2, 48_000)
            .expect("should return a frame from the next buffer, not panic");
        assert_eq!(frame[0], 222, "expected frame from the ahead buffer");
    }
}
