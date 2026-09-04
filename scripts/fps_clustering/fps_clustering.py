"""Thin wrapper kept for backwards compatibility.

The FPS clustering CLI now lives in the installed package and is registered as
the `cryogmm-fps` console script, so it can be run from any directory:

    cryogmm-fps --traj_path ... --traj_top ...

This file simply forwards to it.
"""

from cryogmm.cli.fps_clustering import main

if __name__ == "__main__":
    main()
