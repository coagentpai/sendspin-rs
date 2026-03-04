// ABOUTME: Synced audio player with drift correction (cpal backend)
// ABOUTME: Thin wrapper around AudioRenderer using cpal for audio output

use crate::audio::gain::GainControl;
use crate::audio::renderer::{AudioRenderer, PlaybackQueue, ProcessCallback};
use crate::audio::{AudioBuffer, AudioFormat};
use crate::error::Error;
use crate::sync::ClockSync;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Synced audio output with drift correction (cpal backend).
pub struct SyncedPlayer {
    format: AudioFormat,
    _stream: Stream,
    queue: Arc<Mutex<PlaybackQueue>>,
    /// Last error from the audio stream callback, if any.
    last_error: Arc<Mutex<Option<String>>>,
    gain: GainControl,
}

impl SyncedPlayer {
    /// Create a new synced player using the provided clock sync and optional device.
    ///
    /// The player starts at `volume` (0-100) and `muted` state. These are
    /// applied immediately — the first audio callback uses the correct gain
    /// with no ramp from a default value.
    pub fn new(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        device: Option<Device>,
        volume: u8,
        muted: bool,
    ) -> Result<Self, Error> {
        Self::build(format, clock_sync, device, None, volume, muted)
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
    /// # use sendspin::audio::{AudioFormat, Codec, SyncedPlayer};
    /// # use sendspin::sync::ClockSync;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let format = AudioFormat {
    ///     codec: Codec::Pcm,
    ///     sample_rate: 48_000,
    ///     channels: 2,
    ///     bit_depth: 24,
    ///     codec_header: None,
    /// };
    /// let clock_sync = Arc::new(Mutex::new(ClockSync::new()));
    /// let player = SyncedPlayer::with_process_callback(
    ///     format, clock_sync, None,
    ///     100, false,
    ///     Box::new(|data| { /* e.g. feed a VU meter or visualizer */ }),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_process_callback(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        device: Option<Device>,
        volume: u8,
        muted: bool,
        callback: ProcessCallback,
    ) -> Result<Self, Error> {
        Self::build(format, clock_sync, device, Some(callback), volume, muted)
    }

    fn build(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        device: Option<Device>,
        process_callback: Option<ProcessCallback>,
        volume: u8,
        muted: bool,
    ) -> Result<Self, Error> {
        if format.channels == 0 {
            return Err(Error::Output("channels must be > 0".to_string()));
        }
        let host = cpal::default_host();
        let device = match device {
            Some(device) => device,
            None => host
                .default_output_device()
                .ok_or_else(|| Error::Output("No output device available".to_string()))?,
        };

        let config = StreamConfig {
            channels: format.channels as u16,
            sample_rate: cpal::SampleRate(format.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
        let last_error = Arc::new(Mutex::new(None));
        let gain = GainControl::new(volume, muted);

        let error_clone = Arc::clone(&last_error);

        let stream = Self::build_stream(
            &device,
            &config,
            Arc::clone(&queue),
            clock_sync,
            &format,
            gain.clone(),
            process_callback,
            error_clone,
        )?;
        stream.play().map_err(|e| Error::Output(e.to_string()))?;

        Ok(Self {
            format,
            _stream: stream,
            queue,
            last_error,
            gain,
        })
    }

    /// Enqueue a decoded buffer for playback.
    ///
    /// Scheduling uses `buffer.timestamp` (server time in microseconds) for
    /// drift-corrected playback. The `play_at` field is ignored.
    pub fn enqueue(&self, buffer: AudioBuffer) {
        self.queue.lock().push(buffer);
    }

    /// Clear queued audio and reset playback state.
    pub fn clear(&self) {
        self.queue.lock().clear();
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

    #[allow(clippy::too_many_arguments)]
    fn build_stream(
        device: &Device,
        config: &StreamConfig,
        queue: Arc<Mutex<PlaybackQueue>>,
        clock_sync: Arc<Mutex<ClockSync>>,
        format: &AudioFormat,
        gain_control: GainControl,
        process_callback: Option<ProcessCallback>,
        error_sink: Arc<Mutex<Option<String>>>,
    ) -> Result<Stream, Error> {
        let mut renderer =
            AudioRenderer::new(queue, clock_sync, format, gain_control, process_callback);

        let stream = device
            .build_output_stream(
                config,
                move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                    let cb = Instant::now();
                    let delta = info
                        .timestamp()
                        .playback
                        .duration_since(&info.timestamp().callback)
                        .unwrap_or(Duration::ZERO);
                    renderer.render(data, cb + delta);
                },
                move |err| {
                    eprintln!("Audio stream error: {}", err);
                    *error_sink.lock() = Some(err.to_string());
                },
                None,
            )
            .map_err(|e| Error::Output(e.to_string()))?;

        Ok(stream)
    }
}
