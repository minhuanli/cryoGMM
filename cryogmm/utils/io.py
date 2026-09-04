def save_samples_to_pdb(samples, mdtraj_topology, filename=None):
    """
    Save generated samples as a PDB file.

    Parameters
    ----------
    samples : array-like, shape (Nsamples, n_atoms * n_dim)
        The generated samples to be saved.
    mdtraj_topology : mdtraj.Topology
        An MDTraj Topology object of the molecular system.
    filename : str, optional, default=None
        The output filename with extension (all MDTraj compatible formats).

    Notes
    -----
    If the filename extension is 'pdb', the file will be saved in PDB format.
    Otherwise, the file will be saved in the format specified by the extension.

    Examples
    --------
    >>> save_samples_to_pdb(samples, mdtraj_topology, 'output.pdb')
    """
    import mdtraj as md
    trajectory = md.Trajectory(
        samples.reshape(-1, mdtraj_topology.n_atoms, 3), mdtraj_topology)
    if filename.split('.')[-1] == 'pdb':
        trajectory.save_pdb(filename)
    else:
        trajectory.save(filename)


def load_aligned_trajectory(
    traj_path,
    traj_top,
    alignment_selection=None,
    atom_selection=None,
    reference_frame=0,
    angstrom_to_nm=False,
):
    """Load a trajectory and superpose it onto one of its own frames.

    Convenience loader shared by the clustering and GMM build steps, so both
    operate on identically aligned coordinates.

    Parameters
    ----------
    traj_path : str
        Trajectory coordinates. A `.pt` file holding a tensor of shape
        (N_frames, N_atoms, 3), or any MDTraj-readable trajectory file.
    traj_top : str
        Topology PDB matching the atom order of the trajectory.
    alignment_selection : str, optional
        MDTraj selection string for the superposition, e.g.
        `"(resi > 108) and name BB2"`. If None, no alignment is performed.
    atom_selection : str, optional
        MDTraj selection string to slice after aligning, e.g. `"name BB2"` to
        keep one backbone bead per residue. If None, all atoms are kept.
    reference_frame : int
        Frame all others are superposed onto.
    angstrom_to_nm : bool
        Divide coordinates by 10 for trajectories stored in Angstroms. MDTraj
        and the rest of the pipeline assume nm.

    Returns
    -------
    mdtraj.Trajectory
        The aligned (and optionally sliced) trajectory.

    Examples
    --------
    >>> traj = load_aligned_trajectory(
    ...     "positions_all_traj.pt", "top.pdb",
    ...     alignment_selection="(resi > 108) and name BB2",
    ... )
    """
    import mdtraj as md

    if str(traj_path).endswith(".pt"):
        import torch

        traj_xyz = torch.load(traj_path, weights_only=True).numpy()
        pdb = md.load_pdb(traj_top)
        traj = md.Trajectory(traj_xyz, pdb.top)
    else:
        traj = md.load(traj_path, top=traj_top)

    if angstrom_to_nm:
        traj.xyz = traj.xyz / 10.0

    if alignment_selection is not None:
        traj = traj.superpose(
            traj, reference_frame,
            atom_indices=traj.top.select(alignment_selection),
        )
    if atom_selection is not None:
        traj = traj.atom_slice(traj.top.select(atom_selection))

    return traj
