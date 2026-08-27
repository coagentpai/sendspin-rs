// ABOUTME: Synced audio player with drift correction (PipeWire backend)
// ABOUTME: Native PipeWire stream using AudioRenderer for clock-synced playback

use crate::audio::gain::GainControl;
use crate::audio::renderer::{us_to_ms, AudioRenderer, PlaybackQueue, ProcessCallback};
use crate::audio::synced_player::static_delay_ms_to_us;
use crate::audio::{AudioBuffer, AudioFormat};
use crate::error::Error;
use crate::log_sampling::should_log_sample;
use crate::sync::ClockSync;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use pipewire as pw;
use pw::spa::pod::Pod;
use pw::spa::utils::Direction;
use pw::stream::StreamFlags;

/// Synced audio output with drift correction (PipeWire backend).
///
/// Uses native PipeWire streams instead of going through cpal → ALSA → PipeWire.
/// Shares the same `AudioRenderer` as `SyncedPlayer` for identical drift correction
/// and gain ramping behavior.
pub struct PwSyncedPlayer {
    format: AudioFormat,
    queue: Arc<Mutex<PlaybackQueue>>,
    gain: GainControl,
    last_error: Arc<Mutex<Option<String>>>,
    running: Arc<AtomicBool>,
    reconnect_tx: mpsc::Sender<Duration>,
    thread_handle: Option<JoinHandle<()>>,
    /// Shared with the audio callback. See [`PwSyncedPlayer::set_static_delay`].
    static_delay_us: Arc<AtomicU64>,
}

impl PwSyncedPlayer {
    /// Create a new PipeWire synced player.
    ///
    /// `stream_name` sets the PipeWire node name and description.
    /// If `target_node` is `Some`, the stream routes to that specific node (sink name).
    /// Otherwise PipeWire AUTOCONNECT picks the default.
    pub fn new(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        process_callback: Option<ProcessCallback>,
        volume: u8,
        muted: bool,
        stream_name: &str,
        target_node: Option<String>,
    ) -> Result<Self, Error> {
        if format.channels == 0 {
            return Err(Error::Output("channels must be > 0".to_string()));
        }

        let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
        let gain = GainControl::new(volume, muted);
        let last_error = Arc::new(Mutex::new(None));
        let running = Arc::new(AtomicBool::new(true));

        let static_delay_us = Arc::new(AtomicU64::new(0));

        let thread_queue = Arc::clone(&queue);
        let thread_clock = Arc::clone(&clock_sync);
        let thread_gain = gain.clone();
        let thread_running = Arc::clone(&running);
        let thread_error = Arc::clone(&last_error);
        let thread_format = format.clone();
        let thread_stream_name = stream_name.to_string();
        let thread_static_delay = Arc::clone(&static_delay_us);
        let (reconnect_tx, reconnect_rx) = mpsc::channel();

        let thread_handle = std::thread::Builder::new()
            .name("pw-synced-audio".to_string())
            .spawn(move || {
                if let Err(e) = run_pipewire_loop(
                    thread_format,
                    thread_queue,
                    thread_clock,
                    thread_gain,
                    process_callback,
                    thread_static_delay,
                    thread_running,
                    &thread_stream_name,
                    target_node,
                    reconnect_rx,
                ) {
                    log::error!("PipeWire loop error: {}", e);
                    *thread_error.lock() = Some(e);
                }
            })
            .map_err(|e| Error::Output(format!("Failed to spawn PipeWire thread: {}", e)))?;

        Ok(Self {
            format,
            queue,
            gain,
            last_error,
            running,
            reconnect_tx,
            thread_handle: Some(thread_handle),
            static_delay_us,
        })
    }

    /// Cycle the PipeWire stream: disconnect, stay idle for `idle_gap`, reconnect.
    ///
    /// Disconnecting drops the stream's links so the sink goes idle and (after the
    /// session manager's suspend timeout) releases its device — for Bluetooth sinks
    /// this issues an A2DP SUSPEND, and the following reconnect issues a fresh
    /// A2DP START. Some speakers lock in a bad (deep) playout buffer when the
    /// transport starts while they are still booting; cycling the transport once
    /// they have settled resets it. `idle_gap` must exceed the session manager's
    /// node suspend timeout (WirePlumber default: 5s) for the release to happen.
    ///
    /// Playback is silent during the gap. The renderer re-anchors to the shared
    /// clock on resume, so multi-room sync is re-established automatically.
    pub fn reconnect(&self, idle_gap: Duration) {
        if self.reconnect_tx.send(idle_gap).is_err() {
            log::warn!("PipeWire stream reconnect requested but audio thread is gone");
        }
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

    /// Return the configured audio format.
    pub fn format(&self) -> &AudioFormat {
        &self.format
    }

    /// Check if the audio stream has encountered an error.
    pub fn take_error(&self) -> Option<String> {
        self.last_error.lock().take()
    }

    /// Check if the audio stream has an error without clearing it.
    pub fn has_error(&self) -> bool {
        self.last_error.lock().is_some()
    }

    /// Get a reference to the volume/mute control.
    pub fn gain_control(&self) -> &GainControl {
        &self.gain
    }

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

    /// Set the static playback delay in milliseconds (0–5000, clamped).
    ///
    /// Compensates for downstream latency PipeWire cannot observe (e.g. an
    /// amplifier's DSP): each sample's emission shifts earlier by the same
    /// delay. Takes effect on the next audio callback.
    ///
    /// A delay change is an intentional local timing offset, not clock drift.
    /// Request a one-shot reanchor so the audio callback either skips forward
    /// or waits for the new target time instead of feeding the delay delta
    /// through pitch-shifting drift correction.
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
}

impl Drop for PwSyncedPlayer {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Callback state passed into PipeWire's process callback.
struct PwCallbackState {
    renderer: AudioRenderer,
    channels: u32,
    last_delay_us: u64,
}

/// Run the PipeWire main loop with an audio stream.
#[allow(clippy::too_many_arguments)]
fn run_pipewire_loop(
    format: AudioFormat,
    queue: Arc<Mutex<PlaybackQueue>>,
    clock_sync: Arc<Mutex<ClockSync>>,
    gain_control: GainControl,
    process_callback: Option<ProcessCallback>,
    static_delay_us: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
    stream_name: &str,
    target_node: Option<String>,
    reconnect_rx: mpsc::Receiver<Duration>,
) -> Result<(), String> {
    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None)
        .map_err(|e| format!("Failed to create main loop: {:?}", e))?;
    let context = pw::context::ContextRc::new(&mainloop, None)
        .map_err(|e| format!("Failed to create context: {:?}", e))?;
    let core = context
        .connect_rc(None)
        .map_err(|e| format!("Failed to connect: {:?}", e))?;

    let mut props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_ROLE => "Music",
        *pw::keys::MEDIA_CATEGORY => "Playback",
        *pw::keys::NODE_NAME => stream_name,
        *pw::keys::NODE_DESCRIPTION => stream_name,
    };
    if let Some(ref target) = target_node {
        props.insert("node.target", target.as_str());
        log::info!("PipeWire stream targeting node: {}", target);
    }

    let stream = pw::stream::StreamBox::new(&core, stream_name, props)
        .map_err(|e| format!("Failed to create stream: {:?}", e))?;

    let renderer = AudioRenderer::new(
        queue,
        clock_sync,
        &format,
        gain_control,
        process_callback,
        static_delay_us,
    );

    let state = PwCallbackState {
        renderer,
        channels: format.channels as u32,
        last_delay_us: 0,
    };

    let _listener = stream
        .add_local_listener_with_user_data(state)
        .process(|stream, state| {
            match stream.dequeue_buffer() {
                None => {}
                Some(mut buffer) => {
                    let datas = buffer.datas_mut();
                    if datas.is_empty() {
                        return;
                    }

                    let data = &mut datas[0];
                    let stride = std::mem::size_of::<f32>() * state.channels as usize;

                    if let Some(slice) = data.data() {
                        let n_frames = slice.len() / stride;
                        let n_samples = n_frames * state.channels as usize;

                        // Cast the byte slice to f32 slice
                        let dst: &mut [f32] = unsafe {
                            std::slice::from_raw_parts_mut(
                                slice.as_mut_ptr() as *mut f32,
                                slice.len() / std::mem::size_of::<f32>(),
                            )
                        };

                        // Query PipeWire's downstream sink delay (includes graph
                        // latency + hardware/Bluetooth buffering). RT-safe.
                        let sink_delay = {
                            let mut time: pw::sys::pw_time = unsafe { std::mem::zeroed() };
                            let ret = unsafe {
                                pw::sys::pw_stream_get_time_n(
                                    stream.as_raw_ptr(),
                                    &mut time,
                                    std::mem::size_of::<pw::sys::pw_time>(),
                                )
                            };
                            if ret == 0 && time.rate.denom > 0 && time.delay > 0 {
                                // delay is in graph rate units (samples); convert to Duration
                                Duration::from_nanos(
                                    (time.delay as u64)
                                        .saturating_mul(time.rate.num as u64)
                                        .saturating_mul(1_000_000_000)
                                        / time.rate.denom as u64,
                                )
                            } else {
                                Duration::ZERO
                            }
                        };

                        // Log when sink delay changes significantly (>1ms)
                        let delay_us = sink_delay.as_micros() as u64;
                        let diff = delay_us.abs_diff(state.last_delay_us);
                        if diff > 1000 {
                            log::info!("PipeWire sink delay: {:.1}ms", delay_us as f64 / 1000.0);
                            state.last_delay_us = delay_us;
                        }

                        // The sink delay is this backend's presentation
                        // latency: samples written now play `sink_delay`
                        // later. The renderer shifts its timeline lookup
                        // forward by it, compensating for downstream latency.
                        let dst = &mut dst[..n_samples];
                        state.renderer.render(dst, Instant::now(), sink_delay);

                        let chunk = data.chunk_mut();
                        *chunk.offset_mut() = 0;
                        *chunk.stride_mut() = stride as i32;
                        *chunk.size_mut() = (n_frames * stride) as u32;
                    }
                }
            }
        })
        .register()
        .map_err(|e| format!("Failed to register listener: {:?}", e))?;

    // Build audio format info
    let mut audio_info = pw::spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(pw::spa::param::audio::AudioFormat::F32LE);
    audio_info.set_rate(format.sample_rate);
    audio_info.set_channels(format.channels as u32);

    let mut position = [0u32; pw::spa::param::audio::MAX_CHANNELS];
    if format.channels >= 1 {
        position[0] = pw::spa::sys::SPA_AUDIO_CHANNEL_FL;
    }
    if format.channels >= 2 {
        position[1] = pw::spa::sys::SPA_AUDIO_CHANNEL_FR;
    }
    audio_info.set_position(position);

    let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(pw::spa::pod::Object {
            type_: pw::spa::sys::SPA_TYPE_OBJECT_Format,
            id: pw::spa::sys::SPA_PARAM_EnumFormat,
            properties: audio_info.into(),
        }),
    )
    .map_err(|e| format!("Failed to serialize audio info: {:?}", e))?
    .0
    .into_inner();

    let mut params = [Pod::from_bytes(&values).unwrap()];
    let stream_flags =
        StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS;

    stream
        .connect(Direction::Output, None, stream_flags, &mut params)
        .map_err(|e| format!("Failed to connect stream: {:?}", e))?;

    log::info!(
        "PipeWire synced audio output started: {}Hz {}ch F32LE",
        format.sample_rate,
        format.channels
    );

    // Run main loop with periodic check for stop signal
    while running.load(Ordering::SeqCst) {
        mainloop.loop_().iterate(Duration::from_millis(100));

        if let Ok(idle_gap) = reconnect_rx.try_recv() {
            // Coalesce queued requests into one cycle.
            while reconnect_rx.try_recv().is_ok() {}

            log::info!(
                "Reconnecting PipeWire stream (idle gap {:.1}s)",
                idle_gap.as_secs_f64()
            );
            if let Err(e) = stream.disconnect() {
                log::warn!("PipeWire stream disconnect failed: {:?}", e);
            }

            // Keep iterating the loop while idle so the disconnect is processed
            // and the sink can suspend (releasing its device/transport).
            let deadline = Instant::now() + idle_gap;
            while running.load(Ordering::SeqCst) && Instant::now() < deadline {
                mainloop.loop_().iterate(Duration::from_millis(100));
            }
            if !running.load(Ordering::SeqCst) {
                break;
            }

            let mut params = [Pod::from_bytes(&values).unwrap()];
            match stream.connect(Direction::Output, None, stream_flags, &mut params) {
                Ok(()) => log::info!("PipeWire stream reconnected"),
                Err(e) => {
                    log::error!("PipeWire stream reconnect failed: {:?}", e);
                }
            }
        }
    }

    log::info!("PipeWire synced audio output stopped");
    Ok(())
}
