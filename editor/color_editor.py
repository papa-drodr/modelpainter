import colorsys

import numpy as np

from mesh.color_baker import save_colored_obj


class ColorEditor:
    """
    Per-face color editor for meshes.
    Supports RGB and HSV editing on selected faces.
    """

    def __init__(
        self, vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray
    ):
        """
        Args:
            vertices: mesh vertices (V, 3)
            faces: mesh face indices (F, 3)
            face_colors: RGB color per face (F, 3), values in [0, 1]
        """
        self.vertices = vertices
        self.faces = faces
        self.face_colors = face_colors.copy().astype(np.float32)

    def set_rgb(self, face_indices: list[int], rgb: tuple[float, float, float]):
        """
        Set RGB color for selected faces.

        Args:
            face_indices: list of face indices to edit
            rgb: (R, G, B) values in [0, 1]
        """
        r, g, b = np.clip(rgb, 0.0, 1.0)
        self.face_colors[face_indices] = [r, g, b]

    def set_hsv(self, face_indices: list[int], hsv: tuple[float, float, float]):
        """
        Set HSV color for selected faces.

        Args:
            face_indices: list of face indices to edit
            hsv: (H, S, V) values — H in [0, 360], S and V in [0, 1]
        """
        h, s, v = hsv
        h = h / 360.0  # normalize H to [0, 1] for colorsys
        s = np.clip(s, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        self.face_colors[face_indices] = [r, g, b]

    def shift_hue(self, face_indices: list[int], hue_shift: float):
        """
        Shift hue of selected faces by a given amount.

        Args:
            face_indices: list of face indices to edit
            hue_shift: hue shift in degrees [-360, 360]
        """
        for idx in face_indices:
            r, g, b = self.face_colors[idx]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h = (h + hue_shift / 360.0) % 1.0
            self.face_colors[idx] = colorsys.hsv_to_rgb(h, s, v)

    def adjust_brightness(self, face_indices: list[int], factor: float):
        """
        Multiply brightness (V in HSV) of selected faces by a factor.

        Args:
            face_indices: list of face indices to edit
            factor: brightness multiplier (e.g. 1.5 = brighter, 0.5 = darker)
        """
        for idx in face_indices:
            r, g, b = self.face_colors[idx]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            v = np.clip(v * factor, 0.0, 1.0)
            self.face_colors[idx] = colorsys.hsv_to_rgb(h, s, v)

    def adjust_saturation(self, face_indices: list[int], factor: float):
        """
        Multiply saturation (S in HSV) of selected faces by a factor.

        Args:
            face_indices: list of face indices to edit
            factor: saturation multiplier (e.g. 1.5 = more vivid, 0.0 = grayscale)
        """
        for idx in face_indices:
            r, g, b = self.face_colors[idx]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            s = np.clip(s * factor, 0.0, 1.0)
            self.face_colors[idx] = colorsys.hsv_to_rgb(h, s, v)

    def reset_faces(self, face_indices: list[int], original_colors: np.ndarray):
        """
        Reset selected faces to original baked colors.

        Args:
            face_indices: list of face indices to reset
            original_colors: original face colors (F, 3)
        """
        self.face_colors[face_indices] = original_colors[face_indices]

    def save(self, output_path: str):
        """
        Save edited mesh.

        Args:
            output_path: path to save .obj file
        """
        save_colored_obj(self.vertices, self.faces, self.face_colors, output_path)


if __name__ == "__main__":
    # for test
    import numpy as np

    V = 100
    F = 50
    vertices = np.random.randn(V, 3).astype(np.float32)
    faces = np.random.randint(0, V, (F, 3))
    face_colors = np.random.rand(F, 3).astype(np.float32)

    editor = ColorEditor(vertices, faces, face_colors)

    # set faces 0-9 to red
    editor.set_rgb(list(range(10)), (1.0, 0.0, 0.0))

    # set faces 10-19 to blue via HSV
    editor.set_hsv(list(range(10, 20)), (240.0, 1.0, 1.0))

    # shift hue of faces 20-29
    editor.shift_hue(list(range(20, 30)), hue_shift=90.0)

    editor.save("./output/mesh_edited.obj")
    print("Done.")
