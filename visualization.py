import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from mayavi import mlab
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    view = View(
        Item('scene', editor=SceneEditor(scene_class=MayaviScene), show_label=False),
        resizable=True
    )

    def __init__(self, parent=None):
        super(Visualization, self).__init__()
        self.parent = parent
        self.cylinders = []

    def visualize_points(self, points, vectors=None, draw_style='quiver', cmap='viridis', aspect_ratio=8, draw_box=False, color_by='cmap', color_value=(1, 0, 0), property_array=None, colormap_min=0.0, colormap_max=1.0, tube_length=20, tube_sides=4):
        self.scene.mlab.clf()  # Clear the current figure
        self.cylinders = []  # Clear stored cylinders

        if vectors is None:
            self.scene.mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.1, colormap=cmap)
        else:
            scalars = self.get_scalars(vectors, color_by, color_value, property_array, points, colormap_min, colormap_max)
            if draw_style == 'quiver':
                quiver = self.scene.mlab.quiver3d(points[:, 0], points[:, 1], points[:, 2], vectors[:, 0], vectors[:, 1], vectors[:, 2], scalars=scalars, colormap=cmap, scale_factor=1.0)
                quiver.glyph.color_mode = 'color_by_scalar'
            elif draw_style == 'cylinder':
                for i, (point, vector, scalar) in enumerate(zip(points, vectors, scalars)):
                    if not self.should_draw_cylinder(i, points, 1 / aspect_ratio):
                        continue
                    color = self.normalize_color(scalar) if color_by != 'cmap' else None
                    cylinder = self.draw_cylinder(point, vector, 1 / aspect_ratio, color, cmap, color_by, tube_length, tube_sides)
                    self.cylinders.append(cylinder)

        if color_by in ['P1', 'P2', 'angle']:
            colorbar = self.scene.mlab.colorbar(title=color_by, orientation='vertical')
            if colorbar:
                scalar_bar = colorbar.scalar_bar
                scalar_bar.label_text_property.color = (0, 0, 0)  # Set label color to black
                scalar_bar.title_text_property.color = (0, 0, 0)  # Set title color to black

        if draw_box:
            self.draw_bounding_box(points, vectors)

        self.scene.render()  # Render within the GUI

    def should_draw_cylinder(self, index, points, radius):
        for i, point in enumerate(points):
            if i != index:
                distance = np.linalg.norm(points[index] - point)
                if distance < 1 * radius:
                    return False
        return True

    def draw_cylinder(self, point, vector, radius, color, cmap, color_by, tube_length=20, tube_sides=4):
        length = np.linalg.norm(vector)
        x = np.linspace(point[0], point[0] + vector[0], tube_length)
        y = np.linspace(point[1], point[1] + vector[1], tube_length)
        z = np.linspace(point[2], point[2] + vector[2], tube_length)

        if color_by == 'cmap':
            tube = mlab.plot3d(x, y, z, np.linspace(0, 1, tube_length), tube_radius=radius, tube_sides=tube_sides, colormap=cmap)
        else:
            if isinstance(color, tuple):
                scalars = np.linspace(0, 1, tube_length)
                tube = mlab.plot3d(x, y, z, scalars, tube_radius=radius, tube_sides=tube_sides)
                lut = tube.module_manager.scalar_lut_manager.lut.table.to_array()
                lut[:, :3] = np.array(color) * 255
                tube.module_manager.scalar_lut_manager.lut.table = lut
            else:
                scalars = np.full(tube_length, color)
                tube = mlab.plot3d(x, y, z, scalars, tube_radius=radius, tube_sides=tube_sides, colormap=cmap)
                tube.module_manager.scalar_lut_manager.lut_mode = cmap

        return tube

    def update_cylinder_colors(self, scalars, color_by, color_value, property_array, tube_length=20):
        for i, cylinder in enumerate(self.cylinders):
            scalar = scalars[i]
            if color_by == 'cmap':
                cylinder.mlab_source.scalars = np.linspace(0, 1, tube_length)
                cylinder.module_manager.scalar_lut_manager.lut_mode = color_value
            else:
                color = self.normalize_color(scalar) if color_by != 'cmap' else color_value
                cylinder.mlab_source.color = color

    def normalize_color(self, scalar):
        if isinstance(scalar, (list, tuple)):
            return tuple(max(0, min(1, c)) for c in scalar)
        else:
            return (max(0, min(1, scalar)),) * 3

    def get_scalars(self, vectors, color_by, color_value, property_array, points, colormap_min=0.0, colormap_max=1.0):
        if color_by == 'P1':
            scalars = property_array[0]
        elif color_by == 'P2':
            scalars = property_array[1]
        elif color_by == 'angle':
            scalars = np.degrees(property_array[2])
        elif color_by == 'cmap (x coord)':
            scalars = points[:, 0]
        elif color_by == 'cmap (y coord)':
            scalars = points[:, 1]
        elif color_by == 'cmap (z coord)':
            scalars = points[:, 2]
        elif color_by == 'cmap (x rot)':
            scalars = np.arctan2(vectors[:, 1], vectors[:, 2])
        elif color_by == 'cmap (y rot)':
            scalars = np.arctan2(vectors[:, 0], vectors[:, 2])
        elif color_by == 'cmap (z rot)':
            scalars = np.arctan2(vectors[:, 1], vectors[:, 0])
        elif color_by == 'cmap':
            scalars = np.linspace(0, 1, len(vectors))
        elif color_by == 'rgb':
            scalars = [color_value] * len(vectors)
        else:
            scalars = np.linspace(0, 1, len(vectors))
        
        scalars = np.interp(scalars, (np.min(scalars), np.max(scalars)), (colormap_min, colormap_max))

        return scalars

    def draw_bounding_box(self, points, vectors):
        x_min = np.min(np.concatenate([points[:, 0], points[:, 0] + vectors[:, 0]]))
        x_max = np.max(np.concatenate([points[:, 0], points[:, 0] + vectors[:, 0]]))
        y_min = np.min(np.concatenate([points[:, 1], points[:, 1] + vectors[:, 1]]))
        y_max = np.max(np.concatenate([points[:, 1], points[:, 1] + vectors[:, 1]]))
        z_min = np.min(np.concatenate([points[:, 2], points[:, 2] + vectors[:, 2]]))
        z_max = np.max(np.concatenate([points[:, 2], points[:, 2] + vectors[:, 2]]))

        box_lines = [
            [(x_min, y_min, z_min), (x_max, y_min, z_min)],
            [(x_max, y_min, z_min), (x_max, y_max, z_min)],
            [(x_max, y_max, z_min), (x_min, y_max, z_min)],
            [(x_min, y_max, z_min), (x_min, y_min, z_min)],

            [(x_min, y_min, z_max), (x_max, y_min, z_max)],
            [(x_max, y_min, z_max), (x_max, y_max, z_max)],
            [(x_max, y_max, z_max), (x_min, y_max, z_max)],
            [(x_min, y_max, z_max), (x_min, y_min, z_max)],

            [(x_min, y_min, z_min), (x_min, y_min, z_max)],
            [(x_max, y_min, z_min), (x_max, y_min, z_max)],
            [(x_max, y_max, z_min), (x_max, y_max, z_max)],
            [(x_min, y_max, z_min), (x_min, y_max, z_max)]
        ]

        for line in box_lines:
            self.scene.mlab.plot3d([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]], color=(0, 0, 0), tube_radius=None, line_width=1.0)

    def draw_director(self, director, points):
        print(director)
        center = np.mean(points, axis=0)
        director_length = 2.5  # Set the length for the director for better visualization
        start_point = center - (director * director_length / 2)
        end_point = center + (director * director_length / 2)
        self.scene.mlab.plot3d([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=(1, 0, 0), tube_radius=0.1)

    def draw_plane(self, coordinates, position, axis, tilt_angle=0):
        print(f'Drawing plane at position: {position}, axis: {axis}, tilt_angle: {tilt_angle}')
        if axis == 'x':
            y = np.linspace(np.min(coordinates[:, 1]), np.max(coordinates[:, 1]), 100)
            z = np.linspace(np.min(coordinates[:, 2]), np.max(coordinates[:, 2]), 100)
            y, z = np.meshgrid(y, z)
            x = np.full_like(y, position)
            self.scene.mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=0.3)
        elif axis == 'y':
            x = np.linspace(np.min(coordinates[:, 0]), np.max(coordinates[:, 0]), 100)
            z = np.linspace(np.min(coordinates[:, 2]), np.max(coordinates[:, 2]), 100)
            x, z = np.meshgrid(x, z)
            y = np.full_like(x, position)
            self.scene.mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=0.3)
        elif axis == 'z':
            x = np.linspace(np.min(coordinates[:, 0]), np.max(coordinates[:, 0]), 100)
            y = np.linspace(np.min(coordinates[:, 1]), np.max(coordinates[:, 1]), 100)
            x, y = np.meshgrid(x, y)
            z = np.full_like(x, position)
            if tilt_angle != 0:
                x, y, z = self.apply_tilt(x, y, z, tilt_angle)
            print(f'Drawing mesh with x: {x.shape}, y: {y.shape}, z: {z.shape}')
            self.scene.mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=0.3)

    def apply_tilt(self, x, y, z, tilt_angle):
        tilt_rad = np.radians(tilt_angle)
        x_new = x * np.cos(tilt_rad) - z * np.sin(tilt_rad)
        z_new = x * np.sin(tilt_rad) + z * np.cos(tilt_rad)
        return x_new, y, z_new

class MayaviQWidget(QWidget):
    def __init__(self, parent=None):
        super(MayaviQWidget, self).__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.visualization = Visualization()
        self.scene_widget = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.scene_widget)
        self.setLayout(layout)

    def visualize_points(self, points, vectors=None, draw_style='quiver', cmap='viridis', aspect_ratio=8, draw_box=False, color_by='cmap', color_value=(1, 0, 0), property_array=None, colormap_min=0.0, colormap_max=1.0, tube_length=20, tube_sides=4):
        self.visualization.visualize_points(points, vectors, draw_style, cmap, aspect_ratio, draw_box, color_by, color_value, property_array, colormap_min, colormap_max, tube_length, tube_sides)

    def update_cylinder_colors(self, scalars, color_by, color_value, property_array):
        self.visualization.update_cylinder_colors(scalars, color_by, color_value, property_array)

    def draw_director(self, director, points):
        self.visualization.draw_director(director, points)

    def draw_plane(self, coordinates, position, axis, tilt_angle=0):
        self.visualization.draw_plane(coordinates, position, axis, tilt_angle)
