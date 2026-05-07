import h5py
with h5py.File("act_train_dataset/episode_0.hdf5", "r") as f:
    print("sim attr:", f.attrs["sim"])                          # should print False
    print("action:", f["/action"].shape)                        # should be (T, 8)
    print("qpos:", f["/observations/qpos"].shape)               # should be (T, 8)
    print("cam_static:", f["/observations/images/cam_static"].shape)  # (T, H, W, 3)
    print("cam_wrist:", f["/observations/images/cam_wrist"].shape)    # (T, H, W, 3)