// ABOUTME: Synced audio player with drift correction (PipeWire backend)
// ABOUTME: Native PipeWire stream driving the shared render core

use crate::audio::gain::GainControl;
use crate::audio::synced_player::{PlaybackControl, ProcessCallback};
use crate::audio::{AudioBuffer, AudioFormat};
use crate::error::Error;
use crate::sync::ClockSync;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
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
/// Uses a native PipeWire stream instead of going through cpal → ALSA → PipeWire.
/// It drives the same render core as [`SyncedPlayer`](crate::audio::SyncedPlayer),
/// so drift correction and gain ramping behave identically.
pub struct PwSyncedPlayer {
    control: PlaybackControl,
    running: Arc<AtomicBool>,
    reconnect_tx: mpsc::Sender<Duration>,
    thread_handle: Option<JoinHandle<()>>,
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

        let (control, render) = PlaybackControl::with_render_callback(
            format.clone(),
            clock_sync,
            volume,
            muted,
            process_callback,
        );
        let running = Arc::new(AtomicBool::new(true));

        let thread_control = control.clone();
        let thread_running = Arc::clone(&running);
        let thread_stream_name = stream_name.to_string();
        let (reconnect_tx, reconnect_rx) = mpsc::channel();

        let thread_handle = std::thread::Builder::new()
            .name("pw-synced-audio".to_string())
            .spawn(move || {
                if let Err(e) = run_pipewire_loop(
                    format,
                    render,
                    thread_running,
                    &thread_stream_name,
                    target_node,
                    reconnect_rx,
                ) {
                    log::error!("PipeWire loop error: {}", e);
                    thread_control.fail(e);
                }
            })
            .map_err(|e| Error::Output(format!("Failed to spawn PipeWire thread: {}", e)))?;

        Ok(Self {
            control,
            running,
            reconnect_tx,
            thread_handle: Some(thread_handle),
        })
    }

    /// Creates a cloneable control handle for this player.
    ///
    /// The handle implements [`PlayerOutput`](crate::audio::PlayerOutput), so a
    /// [`PlayerSession`](crate::audio::PlayerSession) can be attached to it. It
    /// reports the PipeWire thread's failure, if it has stopped, from `commit`,
    /// `clear` and the setters.
    pub fn control(&self) -> PlaybackControl {
        self.control.clone()
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
        if let Err(error) = self.control.commit(buffer) {
            log::warn!("Audio buffer rejected: {error}");
        }
    }

    /// Clear queued audio and reset playback state.
    pub fn clear(&self) {
        let _ = self.control.clear();
    }

    /// Return the configured audio format.
    pub fn format(&self) -> &AudioFormat {
        self.control.format()
    }

    /// Check if the audio stream has encountered an error.
    ///
    /// Returns the error message if one occurred, clearing it in the process.
    pub fn take_error(&self) -> Option<String> {
        self.control.take_error()
    }

    /// Check if the audio stream has an error without clearing it.
    pub fn has_error(&self) -> bool {
        self.control.has_error()
    }

    /// Get a reference to the volume/mute control.
    pub fn gain_control(&self) -> &GainControl {
        self.control.gain_control()
    }

    /// Current volume as 0-100.
    pub fn volume(&self) -> u8 {
        self.gain_control().volume()
    }

    /// Whether playback is currently muted.
    pub fn is_muted(&self) -> bool {
        self.gain_control().is_muted()
    }

    /// Set playback volume (0-100).
    pub fn set_volume(&self, volume: u8) {
        self.gain_control().set_volume(volume);
    }

    /// Set mute state.
    pub fn set_mute(&self, muted: bool) {
        self.gain_control().set_mute(muted);
    }

    /// Set the static playback delay in milliseconds (0–5000, clamped).
    ///
    /// Compensates for downstream latency PipeWire cannot observe (e.g. an
    /// amplifier's DSP): each sample's emission shifts earlier by the same
    /// delay. Takes effect on the next audio callback, which reanchors to the
    /// new target time instead of feeding the change through drift
    /// correction. Has no effect once the PipeWire thread has failed.
    pub fn set_static_delay(&self, delay_ms: u16) {
        let _ = self.control.set_static_delay(delay_ms);
    }

    /// Current static delay in milliseconds.
    pub fn static_delay_ms(&self) -> u16 {
        self.control.static_delay_ms()
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
struct PwCallbackState<R> {
    render: R,
    channels: u32,
    last_delay_us: u64,
}

/// Run the PipeWire main loop with an audio stream.
fn run_pipewire_loop(
    format: AudioFormat,
    render: impl FnMut(&mut [f32], Duration) + 'static,
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

    let state = PwCallbackState {
        render,
        channels: format.channels as u32,
        last_delay_us: 0,
    };

    let _listener = stream
        .add_local_listener_with_user_data(state)
        .process(|stream, state| {
            match stream.dequeue_buffer() {
                None => {}
                Some(mut buffer) => {
                    if buffer.datas_mut().is_empty() {
                        return;
                    }

                    let requested = buffer.requested();
                    let datas = buffer.datas_mut();
                    let data = &mut datas[0];
                    let stride = std::mem::size_of::<f32>() * state.channels as usize;

                    if let Some(slice) = data.data() {
                        let n_frames = frames_to_fill(requested, slice.len() / stride);
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
                        // later. The render core shifts its timeline lookup
                        // forward by it, compensating for downstream latency.
                        let dst = &mut dst[..n_samples];
                        (state.render)(dst, sink_delay);

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

/// Frames to fill this cycle.
///
/// `requested` is what the graph asked for (`pw_buffer.requested`); 0 means the
/// server offered no hint, in which case the whole mapped buffer is the only
/// sensible answer.
///
/// Filling the mapped capacity when a request *was* given detaches the callback
/// period from the graph quantum. `pw_stream` only calls `process` again once a
/// buffer recycles, so writing a whole 12288-frame buffer makes the stream wake
/// every 256ms instead of every cycle (21ms at the 1024 quantum this graph
/// runs), inflating both output latency and the dead time in the sync
/// correction loop.
fn frames_to_fill(requested: u64, capacity_frames: usize) -> usize {
    if requested == 0 {
        return capacity_frames;
    }
    // `requested` comes from the server and indexes a raw mapped slice below,
    // so clamp rather than trust it.
    usize::try_from(requested)
        .unwrap_or(capacity_frames)
        .min(capacity_frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::Codec;
    use crate::sync::raw_clock::{Clock, DefaultClock};

    /// The control handle and the render callback share one queue, so audio
    /// committed through the handle is what a backend without a cpal stream
    /// renders.
    #[test]
    fn test_render_callback_plays_committed_audio_once_the_clock_is_synced() {
        let clock = Arc::new(DefaultClock::new());
        let clock_sync = Arc::new(Mutex::new(ClockSync::new(clock.clone())));
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        };
        let (control, mut render) = PlaybackControl::with_render_callback(
            format.clone(),
            Arc::clone(&clock_sync),
            100,
            false,
            None,
        );

        // Ten seconds at a constant level, starting a second ago. The server
        // clock is the client clock here.
        let start_us = clock.now_micros() - 1_000_000;
        for chunk in 0..1000 {
            let buffer = AudioBuffer {
                timestamp: start_us + chunk * 10_000,
                samples: Arc::from(vec![1 << 28; 480 * 2].into_boxed_slice()),
                format: format.clone(),
            };
            control.commit(buffer).expect("matching format");
        }

        let mut block = vec![1.0_f32; 480 * 2];
        let sink_delay = Duration::from_millis(20);
        render(&mut block, sink_delay);
        assert!(
            block.iter().all(|sample| *sample == 0.0),
            "silent until the clock is synchronized"
        );

        for _ in 0..2 {
            let t1 = clock.now_micros();
            clock_sync.lock().update(t1, t1 + 10, t1 + 20, t1 + 30);
        }
        assert!(clock_sync.lock().is_synchronized());

        let played = (0..100).any(|_| {
            render(&mut block, sink_delay);
            block.iter().any(|sample| *sample != 0.0)
        });
        assert!(played, "committed audio reaches the output");
    }

    #[test]
    fn test_fills_only_what_the_graph_requested() {
        // A 1024-frame quantum out of a 12288-frame mapped buffer: writing all
        // 12288 is what stretched the callback period to 256ms.
        assert_eq!(frames_to_fill(1024, 12288), 1024);
    }

    #[test]
    fn test_fills_the_buffer_when_no_request_is_given() {
        // `requested == 0` means the server gave no hint.
        assert_eq!(frames_to_fill(0, 12288), 12288);
    }

    #[test]
    fn test_never_writes_past_the_mapped_buffer() {
        // The server should not ask for more than it mapped, but a request is
        // an untrusted number and this one indexes a raw slice.
        assert_eq!(frames_to_fill(99_999, 1024), 1024);
        assert_eq!(frames_to_fill(u64::MAX, 1024), 1024);
    }
}
