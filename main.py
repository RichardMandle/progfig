import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QCheckBox, QComboBox, QColorDialog, QDialog, QFormLayout
from PyQt5.QtGui import QPixmap, QIcon, QImage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from point_generation import generate_regular_points, hexatic_offset, add_randomness, define_vectors, add_tilt, calculate_average_angle, compute_director
from visualization import MayaviQWidget
from mayavi.core.lut_manager import lut_mode_list

class BlockSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super(BlockSettingsDialog, self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Block Settings')
        layout = QFormLayout()

        self.block_distance_x_input = QLineEdit(self)
        self.block_rotation_x_input = QLineEdit(self)
        self.block_distance_y_input = QLineEdit(self)
        self.block_rotation_y_input = QLineEdit(self)
        self.block_distance_z_input = QLineEdit(self)
        self.block_rotation_z_input = QLineEdit(self)

        layout.addRow('Block Distance X:', self.block_distance_x_input)
        layout.addRow('Block Rotation X:', self.block_rotation_x_input)
        layout.addRow('Block Distance Y:', self.block_distance_y_input)
        layout.addRow('Block Rotation Y:', self.block_rotation_y_input)
        layout.addRow('Block Distance Z:', self.block_distance_z_input)
        layout.addRow('Block Rotation Z:', self.block_rotation_z_input)
        
        self.block_distance_x_input.setText('0.0')
        self.block_rotation_x_input.setText('0.0')
        self.block_distance_y_input.setText('0.0')
        self.block_rotation_y_input.setText('0.0')
        self.block_distance_z_input.setText('0.0')
        self.block_rotation_z_input.setText('0.0')
        
        save_button = QPushButton('Save', self)
        save_button.clicked.connect(self.accept)
        layout.addWidget(save_button)

        self.setLayout(layout)

    def get_block_settings(self):
        return {
            'block_distance_x': float(self.block_distance_x_input.text()),
            'block_rotation_x': float(self.block_rotation_x_input.text()),
            'block_distance_y': float(self.block_distance_y_input.text()),
            'block_rotation_y': float(self.block_rotation_y_input.text()),
            'block_distance_z': float(self.block_distance_z_input.text()),
            'block_rotation_z': float(self.block_rotation_z_input.text())
        }

class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.coordinates = None
        self.vectors = None
        self.previous_options = {}
        self.block_settings = {}

    def initUI(self):
        self.setWindowTitle('Point Generation and Visualization')
        main_layout = QHBoxLayout()
        options_layout = QVBoxLayout()

        self.spacing_label = QLabel('Spacing X:')
        self.spacing_x_input = QLineEdit(self)
        self.spacing_x_input.setText('1.0')

        self.spacing_y_label = QLabel('Spacing Y:')
        self.spacing_y_input = QLineEdit(self)
        self.spacing_y_input.setText('1.0')

        self.spacing_z_label = QLabel('Spacing Z:')
        self.spacing_z_input = QLineEdit(self)
        self.spacing_z_input.setText('1.0')

        self.size_x_label = QLabel('Size X:')
        self.size_x_input = QLineEdit(self)
        self.size_x_input.setText('5.0')

        self.size_y_label = QLabel('Size Y:')
        self.size_y_input = QLineEdit(self)
        self.size_y_input.setText('5.0')

        self.size_z_label = QLabel('Size Z:')
        self.size_z_input = QLineEdit(self)
        self.size_z_input.setText('5.0')

        self.hexatic_checkbox = QCheckBox('In-plane hexatic order?', self)

        self.randomness_x_label = QLabel('Randomness X:')
        self.randomness_x_input = QLineEdit(self)
        self.randomness_x_input.setText('0.0')

        self.randomness_y_label = QLabel('Randomness Y:')
        self.randomness_y_input = QLineEdit(self)
        self.randomness_y_input.setText('0.0')

        self.randomness_z_label = QLabel('Randomness Z:')
        self.randomness_z_input = QLineEdit(self)
        self.randomness_z_input.setText('0.0')

        self.P2_label = QLabel('P2 (Nematic Order Parameter):')
        self.P2_input = QLineEdit(self)
        self.P2_input.setText('0.7')

        self.tilt_angle_label = QLabel('Tilt Angle:')
        self.tilt_angle_input = QLineEdit(self)
        self.tilt_angle_input.setText('0')

        self.polar_checkbox = QCheckBox('Polar?', self)

        self.cmap_label = QLabel('Color Map:')
        self.cmap_dropdown = QComboBox(self)
        valid_colormaps = set(cm.cmap_d.keys())
        available_colormaps = [cmap for cmap in lut_mode_list() if cmap in valid_colormaps]
        for colormap in available_colormaps:
            pixmap = self.create_colormap_preview(colormap)
            self.cmap_dropdown.addItem(QIcon(pixmap), colormap)

        self.draw_style_label = QLabel('Drawing Style:')
        self.draw_style_dropdown = QComboBox(self)
        self.draw_style_dropdown.addItems(['cylinder', 'quiver'])

        self.aspect_ratio_label = QLabel('Aspect Ratio:')
        self.aspect_ratio_input = QLineEdit(self)
        self.aspect_ratio_input.setText('8')

        self.draw_box_checkbox = QCheckBox('Draw Bounding Box?', self)
        self.draw_box_checkbox.setChecked(True)
        
        self.default_params_label = QLabel('Default Parameters:')
        self.default_params_dropdown = QComboBox(self)
        self.default_params_dropdown.addItems(['Default', 'Nematic', 'Smectic A', 'Smectic B', 'Smectic C'])
        self.default_params_dropdown.currentIndexChanged.connect(self.load_default_params)

        self.color_by_label = QLabel('Color By:')
        self.color_by_dropdown = QComboBox(self)
        self.color_by_dropdown.addItems(['cmap', 'P1', 'P2', 'angle', 'rgb'])
        self.color_by_dropdown.currentIndexChanged.connect(self.handle_color_by_change)

        self.color_picker_button = QPushButton('Choose Color', self)
        self.color_picker_button.clicked.connect(self.pick_color)
        self.color_picker_button.setVisible(False)
        self.color_value = (1, 0, 0)

        self.draw_director_checkbox = QCheckBox('Draw Director?', self)
        
        self.draw_layer_planes_checkbox = QCheckBox('Draw Layer Planes?', self)

        self.generate_button = QPushButton('Generate and Draw Points', self)
        self.generate_button.clicked.connect(self.generate_image)

        self.update_button = QPushButton('Update Image', self)
        self.update_button.clicked.connect(self.update_image)
        
        self.block_button = QPushButton('Block Settings', self)
        self.block_button.clicked.connect(self.show_block_settings)

        options_layout.addWidget(self.spacing_label)
        options_layout.addWidget(self.spacing_x_input)
        options_layout.addWidget(self.spacing_y_label)
        options_layout.addWidget(self.spacing_y_input)
        options_layout.addWidget(self.spacing_z_label)
        options_layout.addWidget(self.spacing_z_input)
        options_layout.addWidget(self.size_x_label)
        options_layout.addWidget(self.size_x_input)
        options_layout.addWidget(self.size_y_label)
        options_layout.addWidget(self.size_y_input)
        options_layout.addWidget(self.size_z_label)
        options_layout.addWidget(self.size_z_input)
        options_layout.addWidget(self.hexatic_checkbox)
        options_layout.addWidget(self.randomness_x_label)
        options_layout.addWidget(self.randomness_x_input)
        options_layout.addWidget(self.randomness_y_label)
        options_layout.addWidget(self.randomness_y_input)
        options_layout.addWidget(self.randomness_z_label)
        options_layout.addWidget(self.randomness_z_input)
        options_layout.addWidget(self.P2_label)
        options_layout.addWidget(self.P2_input)
        options_layout.addWidget(self.tilt_angle_label)
        options_layout.addWidget(self.tilt_angle_input)
        options_layout.addWidget(self.polar_checkbox)
        options_layout.addWidget(self.cmap_label)
        options_layout.addWidget(self.cmap_dropdown)
        options_layout.addWidget(self.draw_style_label)
        options_layout.addWidget(self.draw_style_dropdown)
        options_layout.addWidget(self.aspect_ratio_label)
        options_layout.addWidget(self.aspect_ratio_input)
        options_layout.addWidget(self.draw_box_checkbox)
        options_layout.addWidget(self.default_params_label)
        options_layout.addWidget(self.default_params_dropdown)
        options_layout.addWidget(self.color_by_label)
        options_layout.addWidget(self.color_by_dropdown)
        options_layout.addWidget(self.color_picker_button)
        options_layout.addWidget(self.draw_director_checkbox)
        options_layout.addWidget(self.draw_layer_planes_checkbox)
        options_layout.addWidget(self.generate_button)
        options_layout.addWidget(self.update_button)
        options_layout.addWidget(self.block_button)

        self.mayavi_widget = MayaviQWidget(self)
        self.mayavi_widget.visualization.scene.background = (1, 1, 1)
        self.mayavi_widget.setMinimumSize(800, 600)
        main_layout.addLayout(options_layout, 1)
        main_layout.addWidget(self.mayavi_widget, 2)

        self.setLayout(main_layout)

    def create_colormap_preview(self, colormap_name, width=100, height=20):
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack((gradient, gradient))
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(colormap_name))
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        plt.close(fig)
        return QPixmap.fromImage(QImage(image.data, width, height, QImage.Format_RGB888))

    def handle_color_by_change(self):
        if self.color_by_dropdown.currentText() == 'rgb':
            self.color_picker_button.setVisible(True)
        else:
            self.color_picker_button.setVisible(False)

    def pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_value = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)

    def load_default_params(self):
        params = {
            'Nematic': {'spacing_x': '1.1', 'spacing_y': '1.1', 'spacing_z': '1.1', 'size_x': '8', 'size_y': '8', 'size_z': '8', 'hexatic': False, 'randomness_x': '0.3', 'randomness_y': '0.3', 'randomness_z': '0.3', 'P2': '0.7', 'tilt_angle': '0', 'polar': False},
            'Smectic A': {'spacing_x': '1.1', 'spacing_y': '1.1', 'spacing_z': '1.1', 'size_x': '8', 'size_y': '8', 'size_z': '8', 'hexatic': False, 'randomness_x': '0.2', 'randomness_y': '0.2', 'randomness_z': '0.05',  'P2': '0.75', 'tilt_angle': '0', 'polar': False},
            'Smectic B': {'spacing_x': '1.1', 'spacing_y': '1.1', 'spacing_z': '1.1', 'size_x': '8', 'size_y': '8', 'size_z': '8', 'hexatic': True, 'randomness_x': '0.15', 'randomness_y': '0.15', 'randomness_z': '0.025',  'P2': '0.75', 'tilt_angle': '0', 'polar': False},
            'Smectic C': {'spacing_x': '1.1', 'spacing_y': '1.1', 'spacing_z': '1.1', 'size_x': '8', 'size_y': '8', 'size_z': '8', 'hexatic': False, 'randomness_x': '0.2', 'randomness_y': '0.2', 'randomness_z': '0.05', 'P2': '0.75', 'tilt_angle': '22.5', 'polar': False},
        }
        
        selection = self.default_params_dropdown.currentText()
        if selection in params:
            self.spacing_x_input.setText(params[selection]['spacing_x'])
            self.spacing_y_input.setText(params[selection]['spacing_y'])
            self.spacing_z_input.setText(params[selection]['spacing_z'])
            self.size_x_input.setText(params[selection]['size_x'])
            self.size_y_input.setText(params[selection]['size_y'])
            self.size_z_input.setText(params[selection]['size_z'])
            self.hexatic_checkbox.setChecked(params[selection]['hexatic'])
            self.randomness_x_input.setText(params[selection]['randomness_x'])
            self.randomness_y_input.setText(params[selection]['randomness_y'])
            self.randomness_z_input.setText(params[selection]['randomness_z'])
            self.P2_input.setText(params[selection]['P2'])
            self.tilt_angle_input.setText(params[selection]['tilt_angle'])
            self.polar_checkbox.setChecked(params[selection]['polar'])

    def show_block_settings(self):
        dialog = BlockSettingsDialog()
        if dialog.exec_():
            self.block_settings = dialog.get_block_settings()
            print('Block settings updated:', self.block_settings)

    def generate_image(self):
        self.coordinates = None  # Reset coordinates to force regeneration
        self.update_image()

    def update_image(self):
        spacing_x = float(self.spacing_x_input.text())
        spacing_y = float(self.spacing_y_input.text())
        spacing_z = float(self.spacing_z_input.text())
        size_x = float(self.size_x_input.text())
        size_y = float(self.size_y_input.text())
        size_z = float(self.size_z_input.text())
        randomness_x = float(self.randomness_x_input.text())
        randomness_y = float(self.randomness_y_input.text())
        randomness_z = float(self.randomness_z_input.text())
        P2 = float(self.P2_input.text())
        tilt_angle = float(self.tilt_angle_input.text())
        aspect_ratio = float(self.aspect_ratio_input.text())
        cmap = self.cmap_dropdown.currentText()
        draw_style = self.draw_style_dropdown.currentText()
        color_by = self.color_by_dropdown.currentText()
        draw_box = self.draw_box_checkbox.isChecked()
        polar = self.polar_checkbox.isChecked()
        draw_director = self.draw_director_checkbox.isChecked()
        draw_layer_planes = self.draw_layer_planes_checkbox.isChecked()
        
        current_options = {
            'spacing_x': spacing_x,
            'spacing_y': spacing_y,
            'spacing_z': spacing_z,
            'size_x': size_x,
            'size_y': size_y,
            'size_z': size_z,
            'randomness_x': randomness_x,
            'randomness_y': randomness_y,
            'randomness_z': randomness_z,
            'P2': P2,
            'tilt_angle': tilt_angle,
            'block_settings': self.block_settings
        }

        if self.coordinates is None or self.previous_options != current_options:
            spacing = [spacing_x, spacing_y, spacing_z]
            size = [size_x, size_y, size_z]
            self.coordinates = generate_regular_points(spacing, size)
            if self.hexatic_checkbox.isChecked():
                self.coordinates = hexatic_offset(self.coordinates, plane_offset=True)
            self.coordinates = add_randomness(self.coordinates, randomness_x, randomness_y, randomness_z)
            self.vectors = define_vectors(self.coordinates, 1, P2)
            self.coordinates, self.vectors = add_tilt(self.coordinates, self.vectors, tilt_angle)
            self.apply_block_rotations()
                
        self.previous_options = current_options

        P1, P1_array, P2_actual, P2_array, average_angle, angles = calculate_average_angle(self.vectors)
        print(f'P1: {P1}, P2: {P2_actual}')

        if not polar:
            indices = np.random.choice(len(self.vectors), size=len(self.vectors) // 2, replace=False)
            self.coordinates[indices] = self.coordinates[indices] + self.vectors[indices]
            self.vectors[indices] = -self.vectors[indices]

        scalars = self.mayavi_widget.visualization.get_scalars(self.vectors, color_by, self.color_value, np.array([P1_array, P2_array, angles]))
        property_array = np.array([P1_array, P2_array, angles])
        
        # don't worry about recomputing the quiver; its fast. 
        if draw_style == 'quiver':
            self.mayavi_widget.visualize_points(self.coordinates, self.vectors, draw_style, cmap, aspect_ratio, draw_box, color_by, self.color_value, property_array)
        
        # But cylinders, no! reuse the geometry: Ensure visualize_points is called when switching to 'cylinder' or if options change
        if draw_style == 'cylinder' or self.previous_options != current_options:
            self.mayavi_widget.visualize_points(self.coordinates, self.vectors, draw_style, cmap, aspect_ratio, draw_box, color_by, self.color_value, property_array)
        else:
            self.mayavi_widget.update_cylinder_colors(scalars, color_by, self.color_value, property_array)
        
        if draw_layer_planes:
            self.draw_layer_planes(spacing_z, tilt_angle)
                
        if draw_director:
            director = compute_director(self.vectors, polar=polar)
            self.mayavi_widget.visualization.draw_director(director, self.coordinates)

    def apply_block_rotations(self):
        block_distance_x = self.block_settings.get('block_distance_x', 0)
        block_rotation_x = self.block_settings.get('block_rotation_x', 0)
        block_distance_y = self.block_settings.get('block_distance_y', 0)
        block_rotation_y = self.block_settings.get('block_rotation_y', 0)
        block_distance_z = self.block_settings.get('block_distance_z', 0)
        block_rotation_z = self.block_settings.get('block_rotation_z', 0)

        if block_distance_x > 0:
            for i, (point, vector) in enumerate(zip(self.coordinates, self.vectors)):
                step = int(point[0] // block_distance_x)
                angle = step * block_rotation_x
                self.vectors[i] = self.rotate_vector(vector, angle, axis='x')

        if block_distance_y > 0:
            for i, (point, vector) in enumerate(zip(self.coordinates, self.vectors)):
                step = int(point[1] // block_distance_y)
                angle = step * block_rotation_y
                self.vectors[i] = self.rotate_vector(vector, angle, axis='y')

        if block_distance_z > 0:
            for i, (point, vector) in enumerate(zip(self.coordinates, self.vectors)):
                step = int(point[2] // block_distance_z)
                angle = step * block_rotation_z
                self.vectors[i] = self.rotate_vector(vector, angle, axis='z')

    def rotate_vector(self, vector, angle, axis='x'):
        angle_rad = np.radians(angle)
        if axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        elif axis == 'z':
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        return np.dot(rotation_matrix, vector)

    def draw_layer_planes(self, spacing_z, tilt_angle):
        print(f'Drawing layer planes with spacing_z: {spacing_z}, tilt_angle: {tilt_angle}')
        max_z = np.max(self.coordinates[:, 2])
        print(f'max_z: {max_z}')
        for z in np.arange(0, max_z, spacing_z):
            print(f'Drawing plane at z: {z}')
            self.mayavi_widget.draw_plane(self.coordinates, z, 'z', tilt_angle)

    def draw_plane(self, coordinates, position, axis, tilt_angle=0):
        print(f'Drawing plane at position: {position}, axis: {axis}, tilt_angle: {tilt_angle}')
        if axis == 'x':
            y = np.linspace(np.min(coordinates[:, 1]), np.max(coordinates[:, 1]), 100)
            z = np.linspace(np.min(coordinates[:, 2]), np.max(coordinates[:, 2]), 100)
            y, z = np.meshgrid(y, z)
            x = np.full_like(y, position)
            self.mayavi_widget.visualization.scene.mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=0.3)
        elif axis == 'y':
            x = np.linspace(np.min(coordinates[:, 0]), np.max(coordinates[:, 0]), 100)
            z = np.linspace(np.min(coordinates[:, 2]), np.max(coordinates[:, 2]), 100)
            x, z = np.meshgrid(x, z)
            y = np.full_like(x, position)
            self.mayavi_widget.visualization.scene.mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=0.3)
        elif axis == 'z':
            print('zzzzzzz')
            x = np.linspace(np.min(coordinates[:, 0]), np.max(coordinates[:, 0]), 100)
            y = np.linspace(np.min(coordinates[:, 1]), np.max(coordinates[:, 1]), 100)
            x, y = np.meshgrid(x, y)
            z = np.full_like(x, position)
            if tilt_angle != 0:
                x, y, z = self.apply_tilt(x, y, z, tilt_angle)
            print(f'Drawing mesh with x: {x.shape}, y: {y.shape}, z: {z.shape}')
            self.mayavi_widget.visualization.scene.mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=0.3)
            
    def apply_tilt(self, x, y, z, tilt_angle):
        tilt_rad = np.radians(tilt_angle)
        x_new = x * np.cos(tilt_rad) - z * np.sin(tilt_rad)
        z_new = x * np.sin(tilt_rad) + z * np.cos(tilt_rad)
        return x_new, y, z_new
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainApp()
    ex.show()
    sys.exit(app.exec_())
