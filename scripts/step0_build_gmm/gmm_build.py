"""Thin wrapper kept for backwards compatibility.

The GMM build CLI now lives in the installed package and is registered as the
`cryogmm-build-gmm` console script, so it can be run from any directory:

    cryogmm-build-gmm --traj_path ... --cluster_root ...

This file simply forwards to it.
"""

from cryogmm.cli.gmm_build import main

if __name__ == "__main__":
    main()
