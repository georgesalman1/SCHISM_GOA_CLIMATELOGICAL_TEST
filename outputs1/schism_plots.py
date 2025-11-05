#!/usr/bin/env python3
import argparse
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from netCDF4 import num2date
import matplotlib.animation as animation

# -----------------------------
# Filenames (edit these if yours differ)
# -----------------------------
FILE_2D = "out2d_1.nc"
FILE_Z  = "zCoordinates_1.nc"
FILE_S  = "salinity_1.nc"
FILE_T  = "temperature_1.nc"
FILE_R  = "waterDensity_1.nc"
FILE_U  = "horizontalVelX_1.nc"
FILE_V  = "horizontalVelY_1.nc"
FILE_W  = "verticalVelocity_1.nc"

# Wind forcing (eastward/northward 10m wind). Adjust if needed.
FILE_WINDU = "windX_1.nc"
FILE_WINDV = "windY_1.nc"

# global triangulation cache
tri = None


# -----------------------------
# Mesh & helpers
# -----------------------------
def load_mesh():
    g = nc.Dataset(FILE_2D)
    x = g["SCHISM_hgrid_node_x"][:]
    y = g["SCHISM_hgrid_node_y"][:]
    faces = g["SCHISM_hgrid_face_nodes"][:] - 1  # convert 1-based -> 0-based

    tris = []
    for f in faces:
        f = f[f >= 0]  # drop fill values
        if len(f) == 3:
            tris.append(f)
        elif len(f) == 4:
            tris += [[f[0], f[1], f[2]],
                     [f[0], f[2], f[3]]]
        else:
            # generic fan triangulation
            for k in range(1, len(f)-1):
                tris.append([f[0], f[k], f[k+1]])

    tri_local = mtri.Triangulation(x, y, np.asarray(tris, int))
    return g, x, y, tri_local


def get_dry_faces(t):
    """Return boolean mask (True = dry) for triangles at time index t."""
    global tri
    with nc.Dataset(FILE_2D) as ds:
        if "dryFlagNode" not in ds.variables:
            return np.zeros(tri.triangles.shape[0], dtype=bool)
        dn = ds["dryFlagNode"][t, :]
    tri_nodes = tri.triangles
    dry_face = np.any(dn[tri_nodes] > 0.5, axis=1)
    return dry_face


def get_time_from(ds):
    tvar = ds["time"]
    return num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "standard"))


def get_var(ds, names):
    """Pick the first variable in `names` that exists in ds.variables."""
    for n in names:
        if n in ds.variables:
            return ds[n]
    raise KeyError(f"None of {names} found in {ds.filepath()}")


def to_layers_nodes(var, t, nn):
    """
    Return (layers, nodes) array at time t, regardless of original order.
    Supports:
      var[t, layer, node]
      var[t, node, layer]
      var[t, node]
    """
    arr = var[t, ...]
    if arr.ndim == 1:            # (node,)
        return arr[None, :]      # -> (1,node)
    if arr.shape[-1] == nn:      # (layer,node)
        return np.asarray(arr, float)
    elif arr.shape[0] == nn:     # (node,layer)
        return np.asarray(arr, float).T
    else:
        raise ValueError(f"Cannot align node dimension={nn} with {arr.shape}")


def nodes_to_faces(node_vals):
    global tri
    return node_vals[tri.triangles].mean(axis=1)


def global_minmax(var, nn, layers_sel=None):
    """Scan all time for min/max for color scaling."""
    nT = var.shape[0]
    vmin, vmax = np.inf, -np.inf
    for t in range(nT):
        A = to_layers_nodes(var, t, nn)
        if layers_sel is not None:
            A = A[(np.array(layers_sel)-1)]
        vmin = min(vmin, np.nanmin(A))
        vmax = max(vmax, np.nanmax(A))
    return float(vmin), float(vmax)


def get_Z_variable():
    zds = nc.Dataset(FILE_Z)
    Z = get_var(zds, ["zCoordinates","zcor","z_coordinate","z","Z"])
    return zds, Z


def get_depth_sign(Z, nn):
    """Return +1 or -1 so depth is positive downward."""
    Z0 = to_layers_nodes(Z, 0, nn)
    # If model z is negative-down, flip with -1 to make "depth positive down"
    return -1.0 if np.nanmean(Z0) < 0 else 1.0


# -----------------------------
# Plot node map (to pick node index)
# -----------------------------
def plot_node_map(step=1, highlight=None):
    g, x, y, tri_local = load_mesh()
    g.close()
    fig, ax = plt.subplots(figsize=(9,7))
    ax.scatter(x, y, s=6, c="k")
    for i in range(0, len(x), step):
        ax.text(x[i], y[i], str(i+1), fontsize=6, color="tab:blue")  # 1-based label
    if highlight is not None:
        i = highlight-1
        ax.scatter([x[i]], [y[i]], s=60,
                   edgecolor="r", facecolor="none", linewidths=1.5)
        ax.set_title(f"Mesh nodes (highlight node {highlight})")
    else:
        ax.set_title("Mesh nodes (labels are 1-based indices)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.tight_layout(); plt.show()


# -----------------------------
# Vertical profile (e.g. W vs depth)
# -----------------------------
def plot_vertical_profile(node_1based, time_index):
    g, x, y, tri_local = load_mesh()
    g.close()
    nn = len(x)

    zds, Z = get_Z_variable()
    sign = get_depth_sign(Z, nn)

    with nc.Dataset(FILE_W) as wds:
        W = get_var(wds, ["verticalVelocity","w","W"])
        times = get_time_from(wds)
        t = min(time_index, len(times)-1)

        zLN = to_layers_nodes(Z, t, nn) * sign          # (+) downward
        wLN = to_layers_nodes(W, t, nn)

        j = node_1based - 1
        zcol = zLN[:, j]
        wcol = wLN[:, j]

        m = np.isfinite(zcol) & np.isfinite(wcol)

        fig, ax = plt.subplots(figsize=(5,6))
        ax.plot(wcol[m], zcol[m], marker="o", ms=3)
        ax.invert_yaxis()
        ax.set_xlabel("Vertical velocity [m/s]")
        ax.set_ylabel("Depth (+ down) [m]")
        ax.set_title(f"W profile at node {node_1based}  |  t={t}: {times[t]}")
        plt.tight_layout(); plt.show()

    zds.close()


# -----------------------------
# Animate scalar fields (sal, temp, rho, elev)
# Can overlay velocity quiver if 1 panel.
# -----------------------------
def animate_scalar(name, file, var_names, layers, video,
                   top=None, all_layers=False,
                   quiver_step=None, overlay_quiver=False):
    global tri
    g, x, y, tri_local = load_mesh()
    tri = tri_local
    nn = len(x)

    ds = nc.Dataset(file)
    var = get_var(ds, var_names)
    times = get_time_from(ds)
    nT = len(times)

    # layer selection
    if var.ndim == 2:
        # shape (time,node)
        layers_sel = [1]
    else:
        # shape (time,layer,node) OR (time,node,layer)
        Lguess = to_layers_nodes(var, 0, nn).shape[0]
        if all_layers:
            layers_sel = list(range(1, Lguess+1))
        elif top is not None:
            layers_sel = list(range(Lguess-top+1, Lguess+1))
        elif layers:
            layers_sel = layers
        else:
            # default: surface layer (usually last index is surface in SCHISM z-layers)
            layers_sel = [Lguess]

    # color scale over all time/layers selected
    vmin, vmax = global_minmax(var, nn,
                               layers_sel=None if var.ndim==2 else layers_sel)

    # figure layout
    nlay = len(layers_sel)
    ncols = int(np.ceil(np.sqrt(nlay)))
    nrows = int(np.ceil(nlay/ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2*ncols, 3.4*nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()
    pcs = []

    # depth ranges for titles (from Z at t=0)
    if var.ndim == 3:
        zds, Z = get_Z_variable()
        sign = get_depth_sign(Z, nn)
        z0 = to_layers_nodes(Z, 0, nn) * sign  # (+) downward
        depth_ranges = []
        for k in layers_sel:
            zk = z0[k-1, :]
            m = np.isfinite(zk)
            if np.any(m):
                dmin, dmax = float(np.nanmin(zk[m])), float(np.nanmax(zk[m]))
            else:
                dmin, dmax = (np.nan, np.nan)
            depth_ranges.append((dmin, dmax))
        zds.close()
    else:
        depth_ranges = [(np.nan, np.nan)]

    # first-frame dry mask
    dry_mask_faces0 = get_dry_faces(0)

    # first-frame scalar field
    A0 = to_layers_nodes(var, 0, nn)

    for i, k in enumerate(layers_sel):
        node_vals = A0[k-1, :] if var.ndim==3 else A0[0, :]
        face_vals = nodes_to_faces(node_vals)
        face_vals = np.where(dry_mask_faces0, np.nan, face_vals)

        pc = axes[i].tripcolor(tri, face_vals, shading="flat",
                               vmin=vmin, vmax=vmax)
        title = f"Layer {k}"
        if np.isfinite(depth_ranges[i][0]):
            title += f"\n~depth [{depth_ranges[i][0]:.0f},{depth_ranges[i][1]:.0f}] m"
        axes[i].set_title(title, fontsize=8)
        pcs.append(pc)

    for ax in axes[nlay:]:
        ax.axis("off")

    cb = fig.colorbar(pcs[0], ax=axes.tolist(),
                      orientation="horizontal", fraction=0.05, pad=0.06)
    cb.set_label(name)

    ttl = fig.suptitle(f"{name} | t=0: {times[0]}")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # Optional velocity quiver overlay (only if 1 panel)
    if overlay_quiver and (nlay == 1):
        dsU = nc.Dataset(FILE_U)
        dsV = nc.Dataset(FILE_V)
        Uvar = get_var(dsU, ["horizontalVelX","u","U"])
        Vvar = get_var(dsV, ["horizontalVelY","v","V"])

        kq = layers_sel[0]  # same layer index we plotted
        step = max(1, quiver_step if quiver_step else 10)
        idx = np.arange(nn)[::step]

        Ut0 = to_layers_nodes(Uvar, 0, nn)
        Vt0 = to_layers_nodes(Vvar, 0, nn)
        uu0 = Ut0[kq-1, idx] if Uvar.ndim==3 else Ut0[0, idx]
        vv0 = Vt0[kq-1, idx] if Vvar.ndim==3 else Vt0[0, idx]

        qv = axes[0].quiver(x[idx], y[idx], uu0, vv0,
                            scale=5, width=0.0025, color="k")

        def update(frame):
            At = to_layers_nodes(var, frame, nn)
            dry_mask_f = get_dry_faces(frame)

            # update scalar
            for ii, kk in enumerate(layers_sel):
                node_vals_f = At[kk-1, :] if var.ndim==3 else At[0, :]
                face_vals_f = nodes_to_faces(node_vals_f)
                face_vals_f = np.where(dry_mask_f, np.nan, face_vals_f)
                pcs[ii].set_array(face_vals_f)

            # update quiver
            Ut = to_layers_nodes(Uvar, frame, nn)
            Vt = to_layers_nodes(Vvar, frame, nn)
            uu = Ut[kq-1, idx] if Uvar.ndim==3 else Ut[0, idx]
            vv = Vt[kq-1, idx] if Vvar.ndim==3 else Vt[0, idx]
            qv.set_UVC(uu, vv)

            ttl.set_text(f"{name} | t={frame}: {times[frame]}")
            return pcs + [ttl, qv]

        framesN = nT
        if video:
            ani = animation.FuncAnimation(fig, update,
                                          frames=framesN,
                                          interval=250, blit=False)
            try:
                ani.save(video, writer="ffmpeg", dpi=150)
            except Exception:
                from matplotlib.animation import PillowWriter
                ani.save(video.rsplit(".",1)[0]+".gif",
                         writer=PillowWriter(fps=5))
            print(f"✅ wrote {video}")
            plt.close(fig)
        else:
            plt.tight_layout(rect=[0,0,1,0.95])
            plt.show()

        dsU.close(); dsV.close(); ds.close(); g.close()
        return

    # Otherwise (multi-panel or no overlay)
    def update(frame):
        At = to_layers_nodes(var, frame, nn)
        dry_mask_f = get_dry_faces(frame)
        for ii, kk in enumerate(layers_sel):
            node_vals_f = At[kk-1, :] if var.ndim==3 else At[0, :]
            face_vals_f = nodes_to_faces(node_vals_f)
            face_vals_f = np.where(dry_mask_f, np.nan, face_vals_f)
            pcs[ii].set_array(face_vals_f)
        ttl.set_text(f"{name} | t={frame}: {times[frame]}")
        return pcs + [ttl]

    framesN = nT
    if video:
        ani = animation.FuncAnimation(fig, update,
                                      frames=framesN,
                                      interval=250, blit=False)
        try:
            ani.save(video, writer="ffmpeg", dpi=150)
        except Exception:
            from matplotlib.animation import PillowWriter
            ani.save(video.rsplit(".",1)[0]+".gif",
                     writer=PillowWriter(fps=5))
        print(f"✅ wrote {video}")
        plt.close(fig)
    else:
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()

    ds.close(); g.close()


# -----------------------------
# Animate velocity (currents) as quiver
# -----------------------------
def animate_velocity(layers, top, all_layers, quiver_step, video):
    global tri
    g, x, y, tri_local = load_mesh()
    tri = tri_local
    nn = len(x)

    dsU = nc.Dataset(FILE_U); U = get_var(dsU, ["horizontalVelX","u","U"])
    dsV = nc.Dataset(FILE_V); V = get_var(dsV, ["horizontalVelY","v","V"])
    times = get_time_from(dsU)
    nT = len(times)

    if U.ndim == 2:
        layers_sel = [1]  # depth-avg
    else:
        L = to_layers_nodes(U, 0, nn).shape[0]
        if all_layers:
            layers_sel = list(range(1, L+1))
        elif top is not None:
            layers_sel = list(range(L-top+1, L+1))
        elif layers:
            layers_sel = layers
        else:
            layers_sel = [L]  # surface

    step = max(1, quiver_step)
    idx = np.arange(nn)[::step]

    for k in layers_sel:
        fig, ax = plt.subplots(figsize=(8,7))
        Ut = to_layers_nodes(U, 0, nn)
        Vt = to_layers_nodes(V, 0, nn)
        uu = Ut[k-1, idx] if U.ndim==3 else Ut[0, idx]
        vv = Vt[k-1, idx] if V.ndim==3 else Vt[0, idx]
        qv = ax.quiver(x[idx], y[idx], uu, vv,
                       scale=5, width=0.0025, color='k')
        ax.set_title(f"Velocity quiver | layer {k} | t=0: {times[0]}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

        def upd(frame):
            Ut = to_layers_nodes(U, frame, nn)
            Vt = to_layers_nodes(V, frame, nn)
            uu = Ut[k-1, idx] if U.ndim==3 else Ut[0, idx]
            vv = Vt[k-1, idx] if V.ndim==3 else Vt[0, idx]
            qv.set_UVC(uu, vv)
            ax.set_title(f"Velocity quiver | layer {k} | t={frame}: {times[frame]}")
            return (qv,)

        if video:
            ani = animation.FuncAnimation(fig, upd,
                                          frames=nT,
                                          interval=250, blit=False)
            outfile = video if len(layers_sel)==1 else video.replace(".mp4", f"_L{k}.mp4")
            try:
                ani.save(outfile, writer="ffmpeg", dpi=150)
            except Exception:
                from matplotlib.animation import PillowWriter
                ani.save(outfile.replace(".mp4",".gif"),
                         writer=PillowWriter(fps=5))
            print(f"✅ wrote", outfile)
            plt.close(fig)
        else:
            plt.show()

    dsU.close(); dsV.close(); g.close()


# -----------------------------
# Animate wind (quiver of wind forcing)
# -----------------------------
def animate_wind(quiver_step, video):
    """
    Plot wind vectors (eastward, northward) in time.
    We try to read wind from either:
      - separate windX_1.nc / windY_1.nc
      - or out2d_1.nc (if wind fields were dumped there)
    Assumes wind is 2D (time,node), not layered.

    If we cannot find any matching wind variables, we just print a message
    and return without crashing.
    """
    global tri

    # --- load mesh
    g, x, y, tri_local = load_mesh()
    tri = tri_local
    g.close()
    nn = len(x)

    # --- try to open a pair of datasets that actually exist and contain wind vars
    dsWU = None
    dsWV = None
    Uwind = None
    Vwind = None
    times = None

    for candU, candV in WIND_DATASETS:
        try:
            tmpU = nc.Dataset(candU)
        except FileNotFoundError:
            continue
        try:
            tmpV = nc.Dataset(candV)
        except FileNotFoundError:
            tmpU.close()
            continue

        # try to pull a U-wind var
        try:
            Uvar = get_var(tmpU, WIND_U_CANDIDATES)
        except KeyError:
            tmpU.close()
            tmpV.close()
            continue

        # try to pull a V-wind var
        try:
            Vvar = get_var(tmpV, WIND_V_CANDIDATES)
        except KeyError:
            tmpU.close()
            tmpV.close()
            continue

        # try to read time
        try:
            times_here = get_time_from(tmpU)
        except Exception:
            # if time isn't in tmpU, try tmpV
            try:
                times_here = get_time_from(tmpV)
            except Exception:
                tmpU.close()
                tmpV.close()
                continue

        # success path
        dsWU = tmpU
        dsWV = tmpV
        Uwind = Uvar
        Vwind = Vvar
        times = times_here
        break

    # --- if we never succeeded:
    if dsWU is None or dsWV is None or Uwind is None or Vwind is None or times is None:
        print("❌ animate_wind: could not find wind fields in any known file.")
        print("   Check if you wrote wind to NetCDF (e.g. windX_1.nc / windY_1.nc) or into out2d_1.nc.")
        return

    # --- now we can animate
    nT = len(times)
    step = max(1, quiver_step)
    idx = np.arange(nn)[::step]

    # first frame
    U0 = to_layers_nodes(Uwind, 0, nn)  # -> (1,node)
    V0 = to_layers_nodes(Vwind, 0, nn)
    uu0 = U0[0, idx]
    vv0 = V0[0, idx]

    fig, ax = plt.subplots(figsize=(8,7))
    qv = ax.quiver(x[idx], y[idx], uu0, vv0,
                   scale=5, width=0.0025, color='k')
    ax.set_title(f"Wind quiver | t=0: {times[0]}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    def upd(frame):
        Uf = to_layers_nodes(Uwind, frame, nn)
        Vf = to_layers_nodes(Vwind, frame, nn)
        uu = Uf[0, idx]
        vv = Vf[0, idx]
        qv.set_UVC(uu, vv)
        ax.set_title(f"Wind quiver | t={frame}: {times[frame]}")
        return (qv,)

    if video:
        ani = animation.FuncAnimation(fig, upd,
                                      frames=nT,
                                      interval=250,
                                      blit=False)
        try:
            ani.save(video, writer="ffmpeg", dpi=150)
        except Exception:
            from matplotlib.animation import PillowWriter
            ani.save(video.replace(".mp4", ".gif"),
                     writer=PillowWriter(fps=5))
        print(f"✅ wrote", video)
        plt.close(fig)
    else:
        plt.show()

    dsWU.close()
    dsWV.close()

# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="SCHISM quick plotting toolkit")

    p.add_argument("--show-node-map", action="store_true",
                   help="Show mesh with node IDs")

    p.add_argument("--node", type=int,
                   help="1-based node index for vertical profile")

    p.add_argument("--time", type=int, default=0,
                   help="time index for profile or snapshot")

    p.add_argument("--profile", action="store_true",
                   help="Plot vertical velocity profile at node/time")

    p.add_argument("--animate",
                   choices=["salinity","temperature","density","elevation","velocity","wind"],
                   help="Make animation (or show first frame if --video not set)")

    p.add_argument("--layers", type=int, nargs="+",
                   help="1-based layer numbers to plot for scalars/velocity")

    p.add_argument("--top", type=int,
                   help="Plot top N layers (for scalars/velocity)")

    p.add_argument("--all-layers", action="store_true",
                   help="Plot all layers (for scalars)")

    p.add_argument("--video", type=str,
                   help="output video filename (mp4). If omitted, just shows figure.")

    p.add_argument("--quiver-step", type=int, default=10,
                   help="subsample step for quiver arrows (velocity/wind/overlay). Bigger = fewer arrows")

    p.add_argument("--label-step", type=int, default=1,
                   help="label every Nth node in node-map")

    args = p.parse_args()

    # Pre-load mesh once so tri is available early
    global tri
    g0, x0, y0, tri0 = load_mesh()
    tri = tri0
    g0.close()

    if args.show_node_map:
        plot_node_map(step=args.label-step, highlight=args.node)

    if args.profile and args.node:
        plot_vertical_profile(node_1based=args.node, time_index=args.time)

    if args.animate:
        if args.animate == "salinity":
            animate_scalar(
                "Salinity [PSU]",
                FILE_S,
                ["salinity", "S"],
                layers=args.layers,
                video=args.video,
                top=args.top,
                all_layers=args.all_layers,
                quiver_step=args.quiver_step,
                overlay_quiver=True
            )

        elif args.animate == "temperature":
            animate_scalar(
                "Temperature [°C]",
                FILE_T,
                ["temperature","temp","T"],
                layers=args.layers,
                video=args.video,
                top=args.top,
                all_layers=args.all_layers,
                quiver_step=args.quiver_step,
                overlay_quiver=True
            )

        elif args.animate == "density":
            animate_scalar(
                "Density [kg/m³]",
                FILE_R,
                ["waterDensity","rho","density"],
                layers=args.layers,
                video=args.video,
                top=args.top,
                all_layers=args.all_layers,
                quiver_step=args.quiver_step,
                overlay_quiver=False
            )

        elif args.animate == "elevation":
            animate_scalar(
                "Elevation [m]",
                FILE_2D,
                ["elevation","eta"],
                layers=None,
                video=args.video,
                top=None,
                all_layers=False,
                quiver_step=args.quiver_step,
                overlay_quiver=False
            )

        elif args.animate == "velocity":
            animate_velocity(
                layers=args.layers,
                top=args.top,
                all_layers=args.all_layers,
                quiver_step=args.quiver_step,
                video=args.video
            )

        elif args.animate == "wind":
            animate_wind(
                quiver_step=args.quiver_step,
                video=args.video
            )

if __name__ == "__main__":
    main()
