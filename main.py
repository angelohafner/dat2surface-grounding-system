import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import FixedLocator


class PotentialPlotter3D:
    """
    Plot 3D surface from .dat (X,Y,Z) and overlay a grid (segments) from Excel (X1,Y1,X2,Y2).
    Automatically tries transforms (invert / swap axes) to align the grid with the surface area.
    """

    def __init__(
        self,
        dat_path: str,
        mesh_excel_path: str,
        mesh_z_mode: str = "zmin",     # "zmin" or "zero" or numeric string like "10.0"
        mesh_color: str = "black",
        mesh_lw: float = 1.2,
        z_label: str = r"Potencial (V)",
        box_aspect=(0.5, 1.0, 0.25),
        z_offset_for_mesh: float = 0.0,
        verbose: bool = True,
        save_png: bool = True,
        show_plot: bool = True,
        dpi: int = 300
    ):
        self.dat_path = dat_path
        self.mesh_excel_path = mesh_excel_path
        self.mesh_z_mode = mesh_z_mode
        self.mesh_color = mesh_color
        self.mesh_lw = mesh_lw
        self.z_label = z_label
        self.box_aspect = box_aspect
        self.z_offset_for_mesh = z_offset_for_mesh
        self.verbose = verbose
        self.save_png = save_png
        self.show_plot = show_plot
        self.dpi = dpi

        self.df = None
        self.mesh_df = None

    def _read_csv_robust(self, path: str, sep: str) -> pd.DataFrame:
        # Try common encodings used by exports
        encodings = ["utf-16", "utf-8", "latin1"]
        last_exc = None

        for enc in encodings:
            try:
                df = pd.read_csv(path, sep=sep, header=None, engine="python", encoding=enc)
                return df
            except Exception as e:
                last_exc = e

        raise last_exc

    def read_dat_file(self) -> pd.DataFrame:
        # First try TAB (requested), then comma
        try:
            df = self._read_csv_robust(self.dat_path, sep="\t")
        except Exception:
            df = self._read_csv_robust(self.dat_path, sep=",")

        # Convert values to string and clean spaces
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()

        # Replace empty strings with NaN and drop rows with missing values
        df = df.replace("", np.nan)
        df = df.dropna()

        # Convert values to numeric (supports comma decimals)
        for col in df.columns:
            df[col] = df[col].str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()

        self.df = df
        return df

    def read_mesh_excel(self) -> pd.DataFrame:
        dfm = pd.read_excel(self.mesh_excel_path)
        dfm = dfm[["X1", "Y1", "X2", "Y2"]]
        dfm = dfm.fillna(0)

        self.mesh_df = dfm
        return dfm

    def _mesh_z_level(self, z: np.ndarray) -> float:
        if self.mesh_z_mode == "zmin":
            return float(np.nanmin(z))
        if self.mesh_z_mode == "zero":
            return 0.0

        try:
            return float(self.mesh_z_mode)
        except Exception as e:
            raise ValueError("mesh_z_mode must be 'zmin', 'zero', or a numeric value.") from e

    def _bbox_from_segments(self, segs: np.ndarray):
        xs = np.concatenate([segs[:, 0], segs[:, 2]])
        ys = np.concatenate([segs[:, 1], segs[:, 3]])
        return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

    def _overlap_score(self, bbox_a, bbox_b) -> float:
        ax0, ax1, ay0, ay1 = bbox_a
        bx0, bx1, by0, by1 = bbox_b

        ix0 = max(ax0, bx0)
        ix1 = min(ax1, bx1)
        iy0 = max(ay0, by0)
        iy1 = min(ay1, by1)

        if ix1 <= ix0:
            return 0.0
        if iy1 <= iy0:
            return 0.0

        overlap_area = (ix1 - ix0) * (iy1 - iy0)
        surface_area = (ax1 - ax0) * (ay1 - ay0)

        if surface_area <= 0:
            return 0.0

        return float(overlap_area / surface_area)

    def _apply_transform(self, segs: np.ndarray, surface_bbox, mode: str) -> np.ndarray:
        sx0, sx1, sy0, sy1 = surface_bbox

        x1 = segs[:, 0].copy()
        y1 = segs[:, 1].copy()
        x2 = segs[:, 2].copy()
        y2 = segs[:, 3].copy()

        if mode == "xy":
            pass

        elif mode == "xy_mirror_y":
            y1 = (sy0 + sy1) - y1
            y2 = (sy0 + sy1) - y2

        elif mode == "yx":
            tx1 = x1.copy()
            ty1 = y1.copy()
            tx2 = x2.copy()
            ty2 = y2.copy()
            x1 = ty1
            y1 = tx1
            x2 = ty2
            y2 = tx2

        elif mode == "yx_mirror_y":
            tx1 = x1.copy()
            ty1 = y1.copy()
            tx2 = x2.copy()
            ty2 = y2.copy()
            x1 = ty1
            y1 = tx1
            x2 = ty2
            y2 = tx2

            y1 = (sy0 + sy1) - y1
            y2 = (sy0 + sy1) - y2

        else:
            raise ValueError("Unknown transform mode.")

        out = np.column_stack([x1, y1, x2, y2])
        return out

    def _choose_best_transform(self, segs: np.ndarray, surface_bbox):
        candidates = ["xy", "xy_mirror_y", "yx", "yx_mirror_y"]

        best_mode = candidates[0]
        best_score = -1.0
        best_bbox = None

        surface_bbox_tuple = (surface_bbox[0], surface_bbox[1], surface_bbox[2], surface_bbox[3])

        for mode in candidates:
            segs_t = self._apply_transform(segs, surface_bbox_tuple, mode)
            bbox_t = self._bbox_from_segments(segs_t)
            score = self._overlap_score(surface_bbox_tuple, bbox_t)

            if score > best_score:
                best_score = score
                best_mode = mode
                best_bbox = bbox_t

        return best_mode, best_score, best_bbox

    def _title_from_filename_ptbr(self) -> str:
        stem = Path(self.dat_path).stem
        s = stem.replace("_", " ").replace("-", " ").strip()
        s_low = s.lower()

        # Simple mapping based on common ETAP export names
        if "touch" in s_low:
            return "Perfil de potencial de toque"
        if "absolute" in s_low:
            return "Perfil de potencial absoluto"
        if "step" in s_low or "setp" in s_low:
            return "Perfil de tensao de passo"

        # Fallback: use the stem as-is
        return s

    def plot(self) -> None:
        if self.df is None:
            self.read_dat_file()
        if self.mesh_df is None:
            self.read_mesh_excel()

        if self.df.shape[1] < 3:
            raise ValueError("Need at least 3 columns for a surface plot (X, Y, Z).")

        x = self.df.iloc[:, 0].to_numpy()
        y = self.df.iloc[:, 1].to_numpy()
        z = self.df.iloc[:, 2].to_numpy()

        sx0 = float(np.nanmin(x))
        sx1 = float(np.nanmax(x))
        sy0 = float(np.nanmin(y))
        sy1 = float(np.nanmax(y))
        surface_bbox = (sx0, sx1, sy0, sy1)

        segs = self.mesh_df[["X1", "Y1", "X2", "Y2"]].to_numpy(dtype=float)

        best_mode, best_score, best_bbox = self._choose_best_transform(segs, surface_bbox)
        segs_best = self._apply_transform(segs, surface_bbox, best_mode)

        if self.verbose is True:
            print("File:", self.dat_path)
            print("Surface bbox (Xmin,Xmax,Ymin,Ymax):", surface_bbox)
            print("Best mesh transform:", best_mode)
            print("Best overlap score (0..1):", best_score)
            print("Mesh bbox after transform:", best_bbox)

        triang = mtri.Triangulation(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_trisurf(triang, z, cmap="jet")
        fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label=self.z_label)

        ax.set_xlabel(r"$X\,(\rm{m})$")
        ax.set_ylabel(r"$Y\,(\rm{m})$")
        ax.set_box_aspect([self.box_aspect[0], self.box_aspect[1], self.box_aspect[2]])

        title_ptbr = self._title_from_filename_ptbr()
        ax.set_title(title_ptbr)

        z_mesh = self._mesh_z_level(z)
        z_mesh = z_mesh + float(self.z_offset_for_mesh)

        for i in range(segs_best.shape[0]):
            x1 = float(segs_best[i, 0])
            y1 = float(segs_best[i, 1])
            x2 = float(segs_best[i, 2])
            y2 = float(segs_best[i, 3])

            ax.plot(
                [x1, x2],
                [y1, y2],
                [z_mesh, z_mesh],
                color=self.mesh_color,
                linewidth=self.mesh_lw
            )

        ax.set_xlim(sx0, sx1)
        ax.set_ylim(sy0, sy1)

        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))

        zmin_int = int(np.floor(zmin))
        zmax_int = int(np.ceil(zmax))

        zticks = np.linspace(zmin_int, zmax_int, 4)
        zticks = np.round(zticks).astype(int)

        ax.zaxis.set_major_locator(FixedLocator(zticks))
        ax.set_zticklabels([str(v) for v in zticks])

        ax.grid(False)
        plt.tight_layout()

        if self.save_png is True:
            out_png = str(Path(self.dat_path).with_suffix(".png"))
            plt.savefig(out_png, dpi=int(self.dpi), bbox_inches="tight")

        if self.show_plot is True:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":

    mesh_excel_path = r"coordenadas_para_grafico.xlsx"

    dat_files = [
        r"Absolute-Potential-Profile.dat",
        r"Touch-Potential-Profile.dat",
        r"Setp-Potenctial_Profile.dat",
    ]

    for dat_path in dat_files:
        plotter = PotentialPlotter3D(
            dat_path=dat_path,
            mesh_excel_path=mesh_excel_path,
            mesh_z_mode="zmin",
            mesh_color="black",
            mesh_lw=1.2,
            z_label=r"Potencial (V)",
            box_aspect=(0.5, 1.0, 0.25),
            z_offset_for_mesh=0.0,
            verbose=True,
            save_png=True,
            show_plot=True,
            dpi=300
        )

        plotter.plot()
