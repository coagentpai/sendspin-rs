// ABOUTME: Audio types and processing for sendspin-rs
// ABOUTME: Contains Sample type, AudioFormat, Buffer, and codec definitions

/// Audio decoder implementations (PCM, Opus, FLAC)
pub mod decode;
/// Lock-free volume/mute control
pub mod gain;
/// Audio output trait and implementations
pub mod output;
/// Buffer pool for reusing audio sample buffers
pub mod pool;
/// Backend-agnostic audio renderer with drift correction
pub mod renderer;
/// Sync correction planner for drop/insert cadence
pub mod sync_correction;
/// Synced playback helper using output timestamps (cpal backend)
pub mod synced_player;
/// Core audio type definitions (Sample, Codec, AudioFormat, AudioBuffer)
pub mod types;

/// Synced playback helper using native PipeWire streams
#[cfg(feature = "pipewire")]
pub mod pw_synced_player;

pub use gain::GainControl;
pub use output::{AudioOutput, CpalOutput};
pub use pool::BufferPool;
pub use renderer::ProcessCallback;
pub use sync_correction::{CorrectionPlanner, CorrectionSchedule};
pub use synced_player::SyncedPlayer;
pub use types::{AudioBuffer, AudioFormat, Codec, Sample};

#[cfg(feature = "pipewire")]
pub use pw_synced_player::PwSyncedPlayer;
