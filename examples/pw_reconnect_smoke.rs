// ABOUTME: Manual smoke test for PwSyncedPlayer::reconnect()
// ABOUTME: Plays silence to a target sink, cycles the stream, exits
//
// Run with an existing sink (e.g. a null sink created via
//   pw-cli create-node adapter '{ factory.name=support.null-audio-sink
//     node.name=btsim_sink media.class=Audio/Sink object.linger=true
//     audio.position=[FL,FR] }'
// ) and watch links with pw-dump while it runs:
//   cargo run --example pw_reconnect_smoke --features pipewire -- btsim_sink

#[cfg(feature = "pipewire")]
fn main() {
    use parking_lot::Mutex;
    use sendspin::audio::AudioFormat;
    use sendspin::sync::clock::ClockSync;
    use sendspin::{DefaultClock, PwSyncedPlayer};
    use std::sync::Arc;
    use std::time::Duration;

    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .init();

    let target = std::env::args().nth(1);
    println!("target sink: {:?}", target);

    let format = AudioFormat {
        codec: sendspin::audio::Codec::Pcm,
        sample_rate: 48000,
        channels: 2,
        bit_depth: 16,
        codec_header: None,
    };
    let clock_sync = Arc::new(Mutex::new(ClockSync::new(Arc::new(DefaultClock::new()))));

    let player = PwSyncedPlayer::new(
        format,
        clock_sync,
        None,
        100,
        false,
        "pw-reconnect-smoke",
        target,
    )
    .expect("failed to create player");

    println!("phase 1: connected, idling 5s (check links in pw-dump)");
    std::thread::sleep(Duration::from_secs(5));

    println!("phase 2: reconnect with 6s idle gap (links should drop)");
    player.reconnect(Duration::from_secs(6));
    std::thread::sleep(Duration::from_secs(3));
    println!("phase 2b: mid-gap (stream should be unlinked now)");
    std::thread::sleep(Duration::from_secs(5));

    println!("phase 3: reconnected, idling 5s (links should be back)");
    std::thread::sleep(Duration::from_secs(5));

    println!("done");
}

#[cfg(not(feature = "pipewire"))]
fn main() {
    eprintln!("build with --features pipewire");
}
