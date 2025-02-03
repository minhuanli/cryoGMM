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