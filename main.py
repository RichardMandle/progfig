import sys
import json
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QCheckBox, QComboBox, QColorDialog, QGridLayout
from PyQt5.QtGui import QPixmap, QIcon, QImage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from point_generation import generate_regular_points, hexatic_offset, add_randomness, define_vectors, add_tilt, calculate_average_angle, compute_director
from visualization import MayaviQWidget
from mayavi.core.lut_manager import lut_mode_list
from adv_vis_settings import AdvancedVisualizationSettings
from block_settings import BlockSettingsDialog

def get_defaults():
    '''
    Load some default image/phase types from our json file
    '''
    with open('phase_types.json', 'r') as f:
        return json.load(f)
        
class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.default_params = get_defaults()
        self.initUI()
        self.coordinates = None
        self.vectors = None
        self.previous_options = {}
        self.block_settings = {}
        self.adv_vis_settings = {
            'cylinder_resolution': 20,
            'cylinder_sides': 4,
            'aspect_ratio': 8,
            'draw_director': False,
            'draw_layer_planes': False,
            'draw_bounding_box': False
        }

    def initUI(self):
        self.setWindowTitle('Point Generation and Visualization')
        main_layout = QHBoxLayout()
        options_layout = QVBoxLayout()

        spacing_grid = QGridLayout()
        size_grid = QGridLayout()
        randomness_grid = QGridLayout()

        spacing_label = QLabel('Spacing:')
        size_label = QLabel('Size:')
        randomness_label = QLabel('Randomness:')

        self.spacing_x_input = QLineEdit(self)
        self.spacing_x_input.setFixedWidth(30)
        self.spacing_x_input.setText('0.6')
        self.spacing_y_input = QLineEdit(self)
        self.spacing_y_input.setFixedWidth(30)
        self.spacing_y_input.setText('0.6')
        self.spacing_z_input = QLineEdit(self)
        self.spacing_z_input.setFixedWidth(30)
        self.spacing_z_input.setText('1.2')

        self.size_x_input = QLineEdit(self)
        self.size_x_input.setText('5.0')
        self.size_x_input.setFixedWidth(30)
        self.size_y_input = QLineEdit(self)
        self.size_y_input.setText('5.0')
        self.size_y_input.setFixedWidth(30)
        self.size_z_input = QLineEdit(self)
        self.size_z_input.setText('5.0')
        self.size_z_input.setFixedWidth(30)
        
        self.randomness_x_input = QLineEdit(self)
        self.randomness_x_input.setText('0.1')
        self.randomness_x_input.setFixedWidth(30)
        self.randomness_y_input = QLineEdit(self)
        self.randomness_y_input.setText('0.1')
        self.randomness_y_input.setFixedWidth(30)
        self.randomness_z_input = QLineEdit(self)
        self.randomness_z_input.setText('0.1')
        self.randomness_z_input.setFixedWidth(30)

        spacing_grid.addWidget(QLabel('X:'), 0, 0)
        spacing_grid.addWidget(self.spacing_x_input, 0, 1)
        spacing_grid.addWidget(QLabel('Y:'), 0, 2)
        spacing_grid.addWidget(self.spacing_y_input, 0, 3)
        spacing_grid.addWidget(QLabel('Z:'), 0, 4)
        spacing_grid.addWidget(self.spacing_z_input, 0, 5)

        size_grid.addWidget(QLabel('X:'), 0, 0)
        size_grid.addWidget(self.size_x_input, 0, 1)
        size_grid.addWidget(QLabel('Y:'), 0, 2)
        size_grid.addWidget(self.size_y_input, 0, 3)
        size_grid.addWidget(QLabel('Z:'), 0, 4)
        size_grid.addWidget(self.size_z_input, 0, 5)

        randomness_grid.addWidget(QLabel('X:'), 0, 0)
        randomness_grid.addWidget(self.randomness_x_input, 0, 1)
        randomness_grid.addWidget(QLabel('Y:'), 0, 2)
        randomness_grid.addWidget(self.randomness_y_input, 0, 3)
        randomness_grid.addWidget(QLabel('Z:'), 0, 4)
        randomness_grid.addWidget(self.randomness_z_input, 0, 5)

        self.hexatic_checkbox = QCheckBox('In-plane hexatic order?', self)

        self.P2_label = QLabel('P2 (Nematic Order Parameter):')
        self.P2_input = QLineEdit(self)
        self.P2_input.setText('1.0')
        
        #tilt
        self.tilt_angle_x_label = QLabel('Tilt Angle X:')
        self.tilt_angle_x_input = QLineEdit(self)
        self.tilt_angle_x_input.setText('0')
        self.tilt_angle_y_label = QLabel('Tilt Angle Y:')
        self.tilt_angle_y_input = QLineEdit(self)
        self.tilt_angle_y_input.setText('0')
        
        #splay
        self.splay_angle_x_label = QLabel('Splay Angle X:')
        self.splay_angle_x_input = QLineEdit(self)
        self.splay_angle_x_input.setText('0')
        self.splay_period_x_label = QLabel('Splay Period X:')
        self.splay_period_x_input = QLineEdit(self)
        self.splay_period_x_input.setText('10')
        self.splay_angle_y_label = QLabel('Splay Angle Y:')
        self.splay_angle_y_input = QLineEdit(self)
        self.splay_angle_y_input.setText('0')
        self.splay_period_y_label = QLabel('Splay Period Y:')
        self.splay_period_y_input = QLineEdit(self)
        self.splay_period_y_input.setText('10')
        self.splay_positive_x_checkbox = QCheckBox('Positive Splay X', self)
        self.splay_positive_y_checkbox = QCheckBox('Positive Splay Y', self)

        self.polar_checkbox = QCheckBox('Polar?', self)

        self.cmap_label = QLabel('Color Map:')
        self.cmap_dropdown = QComboBox(self)
        valid_colormaps = set(dir(cm)) # seemingly, just dir the module? #set(cm.cmap_d.keys())
        available_colormaps = [cmap for cmap in lut_mode_list() if cmap in valid_colormaps]
        for colormap in available_colormaps:
            pixmap = self.create_colormap_preview(colormap)
            self.cmap_dropdown.addItem(QIcon(pixmap), colormap)

        self.draw_style_label = QLabel('Drawing Style:')
        self.draw_style_dropdown = QComboBox(self)
        self.draw_style_dropdown.addItems(['quiver', 'cylinder'])

        self.default_params_label = QLabel('Default Parameters:')
        self.default_params_dropdown = QComboBox(self)
        self.default_params_dropdown.addItems(self.default_params.keys())
        self.default_params_dropdown.currentIndexChanged.connect(self.load_default_params)

        self.color_by_label = QLabel('Color By:')
        self.color_by_dropdown = QComboBox(self)
        self.color_by_dropdown.addItems([
            'cmap', 'cmap (x coord)', 'cmap (y coord)', 'cmap (z coord)', 
            'cmap (x rot)', 'cmap (y rot)', 'cmap (z rot)', 
            'P1', 'P2', 'angle', 'rgb'
        ])
        self.color_by_dropdown.currentIndexChanged.connect(self.handle_color_by_change)

        self.color_picker_button = QPushButton('Choose Color', self)
        self.color_picker_button.clicked.connect(self.pick_color)
        self.color_picker_button.setVisible(False)
        self.color_value = (1, 0, 0)

        self.generate_button = QPushButton('Generate and Draw Points', self)
        self.generate_button.clicked.connect(self.generate_image)

        self.update_button = QPushButton('Update Image', self)
        self.update_button.clicked.connect(self.update_image)
        
        self.block_button = QPushButton('Block Settings', self)
        self.block_button.clicked.connect(self.show_block_settings)

        self.draw_settings_button = QPushButton('Draw Settings', self)
        self.draw_settings_button.clicked.connect(self.show_draw_settings)

        options_layout.addWidget(spacing_label)
        options_layout.addLayout(spacing_grid)
        options_layout.addWidget(size_label)
        options_layout.addLayout(size_grid)
        options_layout.addWidget(randomness_label)
        options_layout.addLayout(randomness_grid)
        options_layout.addWidget(self.hexatic_checkbox)
        options_layout.addWidget(self.P2_label)
        options_layout.addWidget(self.P2_input)
        options_layout.addWidget(self.tilt_angle_x_label)
        options_layout.addWidget(self.tilt_angle_x_input)
        options_layout.addWidget(self.tilt_angle_y_label)
        options_layout.addWidget(self.tilt_angle_y_input)
        
        #new splay options
        options_layout.addWidget(self.splay_angle_x_label)
        options_layout.addWidget(self.splay_angle_x_input)
        options_layout.addWidget(self.splay_period_x_label)
        options_layout.addWidget(self.splay_period_x_input)
        options_layout.addWidget(self.splay_positive_x_checkbox)
        options_layout.addWidget(self.splay_angle_y_label)
        options_layout.addWidget(self.splay_angle_y_input)
        options_layout.addWidget(self.splay_period_y_label)
        options_layout.addWidget(self.splay_period_y_input)
        options_layout.addWidget(self.splay_positive_y_checkbox)

        options_layout.addWidget(self.polar_checkbox)
        options_layout.addWidget(self.cmap_label)
        options_layout.addWidget(self.cmap_dropdown)
        options_layout.addWidget(self.draw_style_label)
        options_layout.addWidget(self.draw_style_dropdown)
        options_layout.addWidget(self.default_params_label)
        options_layout.addWidget(self.default_params_dropdown)
        options_layout.addWidget(self.color_by_label)
        options_layout.addWidget(self.color_by_dropdown)
        options_layout.addWidget(self.color_picker_button)
        options_layout.addWidget(self.generate_button)
        options_layout.addWidget(self.update_button)
        options_layout.addWidget(self.block_button)
        options_layout.addWidget(self.draw_settings_button)

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
        selection = self.default_params_dropdown.currentText()
        if selection in self.default_params:
            params = self.default_params[selection]
            self.spacing_x_input.setText(params['spacing_x'])
            self.spacing_y_input.setText(params['spacing_y'])
            self.spacing_z_input.setText(params['spacing_z'])
            self.size_x_input.setText(params['size_x'])
            self.size_y_input.setText(params['size_y'])
            self.size_z_input.setText(params['size_z'])
            self.hexatic_checkbox.setChecked(params['hexatic'])
            self.randomness_x_input.setText(params['randomness_x'])
            self.randomness_y_input.setText(params['randomness_y'])
            self.randomness_z_input.setText(params['randomness_z'])
            self.P2_input.setText(params['P2'])
            self.tilt_angle_x_input.setText(params['tilt_angle_x'])
            self.tilt_angle_y_input.setText(params['tilt_angle_y'])
            self.polar_checkbox.setChecked(params['polar'])

            blocks = params.get('blocks', {})
            self.block_settings = {
                'block_distance_x': float(blocks.get('block_distance_x', 0)),
                'block_rotation_x': float(blocks.get('block_rotation_x', 0)),
                'block_distance_y': float(blocks.get('block_distance_y', 0)),
                'block_rotation_y': float(blocks.get('block_rotation_y', 0)),
                'block_distance_z': float(blocks.get('block_distance_z', 0)),
                'block_rotation_z': float(blocks.get('block_rotation_z', 0)),
                'draw_x_block_plane': blocks.get('draw_x_block_plane', False),
                'draw_y_block_plane': blocks.get('draw_y_block_plane', False),
                'draw_z_block_plane': blocks.get('draw_z_block_plane', False)
            }

    def show_block_settings(self):
        dialog = BlockSettingsDialog()
        if dialog.exec_():
            self.block_settings = dialog.get_block_settings()
            print('Block settings updated:', self.block_settings)

    def show_draw_settings(self):
        dialog = AdvancedVisualizationSettings(self.adv_vis_settings)
        if dialog.exec_():
            self.adv_vis_settings = dialog.get_settings()
            print('Draw settings updated:', self.adv_vis_settings)

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
        tilt_angle_x = float(self.tilt_angle_x_input.text())
        tilt_angle_y = float(self.tilt_angle_y_input.text())
        cmap = self.cmap_dropdown.currentText()
        draw_style = self.draw_style_dropdown.currentText()
        color_by = self.color_by_dropdown.currentText()
        draw_director = self.adv_vis_settings['draw_director']
        draw_layer_planes = self.adv_vis_settings['draw_layer_planes']
        draw_box = self.adv_vis_settings['draw_bounding_box']
        aspect_ratio = float(self.adv_vis_settings['aspect_ratio'])
        colormap_min = self.adv_vis_settings.get('colormap_min', 0.0)
        colormap_max = self.adv_vis_settings.get('colormap_max', 1.0)
        polar = self.polar_checkbox.isChecked()

        # Splay parameters
        splay_angle_x = float(self.splay_angle_x_input.text())
        splay_angle_y = float(self.splay_angle_y_input.text())
        positive_splay_x = self.splay_positive_x_checkbox.isChecked()
        positive_splay_y = self.splay_positive_y_checkbox.isChecked()
        splay_period_x = float(self.splay_period_x_input.text())
        splay_period_y = float(self.splay_period_y_input.text())

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
            'tilt_angle_x': tilt_angle_x,
            'tilt_angle_y': tilt_angle_y,
            'splay_angle_x': splay_angle_x,
            'splay_angle_y': splay_angle_y,
            'positive_splay_x': positive_splay_x,
            'positive_splay_y': positive_splay_y,
            'splay_period_x': splay_period_x,
            'splay_period_y': splay_period_y,
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
            self.vectors, self.coordinates = self.apply_splay(self.coordinates, self.vectors, splay_angle_x, splay_angle_y, positive_splay_x, positive_splay_y, splay_period_x, splay_period_y)
            self.coordinates, self.vectors = add_tilt(self.coordinates, self.vectors, tilt_angle_x, tilt_angle_y)
            self.apply_block_rotations()
            
        if not polar:
            indices = np.random.choice(len(self.vectors), size=len(self.vectors) // 2, replace=False)
            self.coordinates[indices] = self.coordinates[indices] + self.vectors[indices]
            self.vectors[indices] = -self.vectors[indices]
        
        P1, P1_array, P2_actual, P2_array, average_angle, angles = calculate_average_angle(self.vectors)
        print(f'P1: {P1}, P2: {P2_actual}')
        
        property_array = np.array([P1_array, P2_array, angles])

        if draw_style == 'quiver':
            self.mayavi_widget.visualize_points(self.coordinates, self.vectors, draw_style, cmap, aspect_ratio, draw_box, color_by, self.color_value, property_array, colormap_min, colormap_max)
        elif draw_style == 'cylinder' or self.previous_options != current_options:
            self.mayavi_widget.visualize_points(self.coordinates, self.vectors, draw_style, cmap, aspect_ratio, draw_box, color_by, self.color_value, property_array,  colormap_min, colormap_max,  tube_length=self.adv_vis_settings['cylinder_resolution'], tube_sides=self.adv_vis_settings['cylinder_sides'])
        else:
            self.mayavi_widget.update_cylinder_colors(scalars, color_by, self.color_value, property_array)
            
        if draw_layer_planes:
            self.draw_layer_planes(spacing_z)
                
        if draw_director:
            director = compute_director(self.vectors)
            self.mayavi_widget.visualization.draw_director(director, self.coordinates)
        
        self.draw_block_planes()
        self.previous_options = current_options # lastly, update the options list
        
    def draw_block_planes(self):
        block_distance_x = self.block_settings.get('block_distance_x', 0)
        block_distance_y = self.block_settings.get('block_distance_y', 0)
        block_distance_z = self.block_settings.get('block_distance_z', 0)

        if self.block_settings.get('draw_x_block_plane', False) and block_distance_x > 0:
            max_x = np.max(self.coordinates[:, 0])
            for x in np.arange(0, max_x, block_distance_x):
                self.mayavi_widget.visualization.draw_plane(self.coordinates, x, 'x')

        if self.block_settings.get('draw_y_block_plane', False) and block_distance_y > 0:
            max_y = np.max(self.coordinates[:, 1])
            for y in np.arange(0, max_y, block_distance_y):
                self.mayavi_widget.visualization.draw_plane(self.coordinates, y, 'y')

        if self.block_settings.get('draw_z_block_plane', False) and block_distance_z > 0:
            max_z = np.max(self.coordinates[:, 2])
            for z in np.arange(0, max_z, block_distance_z):
                self.mayavi_widget.visualization.draw_plane(self.coordinates, z, 'z')
                
    def apply_block_rotations(self):
        block_distance_x = float(self.block_settings.get('block_distance_x', 0))
        block_rotation_x = float(self.block_settings.get('block_rotation_x', 0))
        block_distance_y = float(self.block_settings.get('block_distance_y', 0))
        block_rotation_y = float(self.block_settings.get('block_rotation_y', 0))
        block_distance_z = float(self.block_settings.get('block_distance_z', 0))
        block_rotation_z = float(self.block_settings.get('block_rotation_z', 0))

        if block_distance_x > 0:
            for i, (point, vector) in enumerate(zip(self.coordinates, self.vectors)):
                step = int(point[0] // block_distance_x)
                angle = step * block_rotation_x
                center_x = (step + 0.5) * block_distance_x
                self.coordinates[i] = point - np.array([0, 0, np.cos(np.radians(angle))/2])
                rotated_vector = self.rotate_vector(vector, angle, axis='x')
                self.vectors[i] = rotated_vector

        if block_distance_y > 0:
            for i, (point, vector) in enumerate(zip(self.coordinates, self.vectors)):
                step = int(point[1] // block_distance_y)
                angle = step * block_rotation_y
                center_y = (step + 0.5) * block_distance_y
                self.coordinates[i] = point - np.array([0, 0, np.cos(np.radians(angle))/2])
                rotated_vector = self.rotate_vector(vector, angle, axis='y')
                self.vectors[i] = rotated_vector

        if block_distance_z > 0:
            for i, (point, vector) in enumerate(zip(self.coordinates, self.vectors)):
                step = int(point[2] // block_distance_z)
                angle = step * block_rotation_z
                center_z = (step + 0.5) * block_distance_z
                rotated_vector = self.rotate_vector(vector, angle, axis='z')
                self.vectors[i] = rotated_vector

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

    def draw_layer_planes(self, spacing_z):
        block_distance_x = self.block_settings.get('block_distance_x', 0)
        block_rotation_x = self.block_settings.get('block_rotation_x', 0)
        block_distance_y = self.block_settings.get('block_distance_y', 0)
        block_rotation_y = self.block_settings.get('block_rotation_y', 0)
        block_distance_z = self.block_settings.get('block_distance_z', 0)
        block_rotation_z = self.block_settings.get('block_rotation_z', 0)

        if block_distance_z == 0:
            self.draw_planes_in_range(0, np.max(self.coordinates[:, 2]), spacing_z, 0)
        else:
            max_z = np.max(self.coordinates[:, 2])
            for z_start in np.arange(0, max_z, block_distance_z):
                z_end = min(z_start + block_distance_z, max_z)
                angle_z = block_rotation_z * (z_start // block_distance_z)
                angle_z = 0 # this seems to improve matters.
                self.draw_planes_in_range(z_start, z_end, spacing_z, angle_z)

    def draw_planes_in_range(self, z_start, z_end, spacing_z, block_rotation_z):
        for z in np.arange(z_start, z_end, spacing_z):
            self.draw_plane(z, 'z', block_rotation_z)

    def draw_plane(self, position, axis, block_rotation_z=0):
        if axis == 'z':
            x = np.linspace(np.min(self.coordinates[:, 0]), np.max(self.coordinates[:, 0]), 100)
            y = np.linspace(np.min(self.coordinates[:, 1]), np.max(self.coordinates[:, 1]), 100)
            x, y = np.meshgrid(x, y)
            z = np.full_like(x, position)
            if block_rotation_z != 0:
                x, y, z = self.apply_tilt(x, y, z, 0, 0, block_rotation_z)
            self.mayavi_widget.visualization.scene.mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=0.3)

    def apply_tilt(self, x, y, z, tilt_angle_x=0, tilt_angle_y=0, tilt_angle_z=0):
        if tilt_angle_z != 0:
            tilt_rad_z = np.radians(tilt_angle_z)
            x_new = x * np.cos(tilt_rad_z) - y * np.sin(tilt_rad_z)
            y_new = x * np.sin(tilt_rad_z) + y * np.cos(tilt_rad_z)
            x, y = x_new, y_new
        return x, y, z
    
    # might want to look at moving the coordinates so that the splayed / rotated vectors don't overlap with neighbors with alternate splay direction.
    # needs a logical chek to take place (why is all this code in main anyway?)
    def apply_splay(self, coordinates, vectors, splay_angle_x, splay_angle_y, positive_splay_x, positive_splay_y, splay_period_x, splay_period_y):
        for i, (coord, vec) in enumerate(zip(coordinates, vectors)):
            original_coord = coord.copy()

            if splay_angle_x != 0 and splay_period_x > 0:
                offset_x = (coord[0] % splay_period_x) / splay_period_x  # Normalize to range [0, 1)
                if positive_splay_x:
                    angle = (-0.5 + offset_x) * 2 * splay_angle_x  # Vary from -splay_angle_x to +splay_angle_x
                    coordinates[i][0] -= np.sin(np.radians(angle))
                else:
                    angle = (0.5 - offset_x) * 2 * splay_angle_x  # Vary from +splay_angle_x to -splay_angle_x
                    coordinates[i][0] += np.sin(np.radians(angle))
                vectors[i] = self.rotate_vector(vec, angle, 'y')

            if splay_angle_y != 0 and splay_period_y > 0:
                offset_y = (coord[1] % splay_period_y) / splay_period_y  # Normalize to range [0, 1)
                if positive_splay_y:
                    angle = (-0.5 + offset_y) * 2 * splay_angle_y  # Vary from -splay_angle_y to +splay_angle_y
                    coordinates[i][1] -= np.sin(np.radians(angle))
                else:
                    angle = (0.5 - offset_y) * 2 * splay_angle_y  # Vary from +splay_angle_y to -splay_angle_y
                    coordinates[i][1] += np.sin(np.radians(angle))
                vectors[i] = self.rotate_vector(vec, angle, 'x')

        return vectors, coordinates
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainApp()
    ex.show()
    sys.exit(app.exec_())