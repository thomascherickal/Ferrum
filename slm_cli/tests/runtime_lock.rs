//! The cross-process runtime lock: while held, other holders (even in this process, via a fresh
//! file description) are blocked; it is released on drop.
mod common;

#[test]
fn runtime_lock_excludes_releases_and_survives_panics() {
    let guard = common::runtime_lock();
    assert!(
        common::runtime_lock_is_held_elsewhere(),
        "a second holder must see the lock taken"
    );
    // a blocked acquirer proceeds only after we drop
    let t = std::thread::spawn(|| {
        let start = std::time::Instant::now();
        let _g = common::runtime_lock();
        start.elapsed()
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    drop(guard);
    let waited = t.join().unwrap();
    assert!(
        waited >= std::time::Duration::from_millis(250),
        "{waited:?}"
    );
    assert!(
        !common::runtime_lock_is_held_elsewhere(),
        "released after both drops"
    );

    // a panicking holder neither deadlocks nor poisons the lock
    let r = std::panic::catch_unwind(|| {
        let _g = common::runtime_lock();
        panic!("holder dies");
    });
    assert!(r.is_err());
    let _g = common::runtime_lock(); // must not deadlock or poison
}
