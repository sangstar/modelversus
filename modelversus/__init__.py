try:
    import modelversus._rust as _rust
except ModuleNotFoundError:
    # For some reason maturin may insist on placing
    # .so files in its own dir in site=packages/
    # for MacOS users, so try this:
    from _rust import _rust
