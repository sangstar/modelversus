try:
    import modelversus._rust as _rust
except ModuleNotFoundError:
    # For some reason maturin may insist on placing
    # .so files in their own dirs in site_packages/
    # for MacOS users, so try this in case
    from _rust import _rust

from _rust import score_batch

__all__ = [score_batch]