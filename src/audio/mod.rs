// ABOUTME: Audio types and processing for sendspin-rs
// ABOUTME: Contains Sample type, AudioFormat, Buffer, and codec definitions

/// Audio decoder implementations (PCM, Opus, FLAC)
pub mod decode;
mod diagnostics;
/// Lock-free volume/mute control
pub mod gain;
/// Buffer pool for reusing audio sample buffers
pub mod pool;
/// Sync correction planner for drop/insert cadence
pub mod sync_correction;
/// Synced playback helper using output timestamps (cpal backend)
pub mod synced_player;
/// Core audio type definitions (Sample, Codec, AudioFormat, AudioBuffer)
pub mod types;

/// Synced playback helper using native PipeWire streams
#[cfg(feature = "pipewire")]
pub mod pw_synced_player;

pub(crate) mod player_api;
mod player_session;

pub use crate::protocol::messages::StreamPlayerConfig;
pub use gain::GainControl;
pub use player_api::{
    AudioWork, OutputInstallPermit, PlayerEvent, PlayerOutput, PlayerReceiver, PlayerSession,
};
pub use player_session::{
    AppliedControls, CommitRejected, ControlUpdate, InstallRejected, OutputError, PlayerStatus,
};
pub use pool::BufferPool;
pub use sync_correction::{CorrectionPlanner, CorrectionSchedule};
pub use synced_player::PlaybackControl;
pub use synced_player::{ProcessCallback, SyncedPlayer, SyncedPlayerConfig};
pub use types::{AudioBuffer, AudioFormat, Codec};

#[cfg(feature = "pipewire")]
pub use pw_synced_player::PwSyncedPlayer;
