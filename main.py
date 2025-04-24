import sys
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QCheckBox,
    QComboBox, QColorDialog, QGridLayout
)
from PyQt5.QtGui import QPixmap, QIcon, QImage
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from point_generation import (
    generate_regular_points, hexatic_offset,
    add_randomness, define_vectors,
    add_tilt, calculate_average_angle,
    compute_director
)
from visualization import MayaviQWidget
from mayavi.core.lut_manager import lut_mode_list
from adv_vis_settings import AdvancedVisualizationSettings
from block_settings import BlockSettingsDialog

'''
Progfig - a tool for programatically generating images of liquid crystal phase types.
Author  - Dr. Richard Mandle, UoL, 2024 -->

Why? Sometimes we want images of phase types, structures etc which are free from bias or whatever
     So, having a tool that builds these by considering things like particle density, vector length, order paramters
     frees us from having to make design choices ourselves. Also, it looks nice (mostly)
'''

def get_defaults():
    """Load default phase settings from JSON."""
    with open('phase_types.json', 'r') as f:
        return json.load(f)

class MainApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load defaults
        self.default_params = get_defaults()

        # State
        self.coordinates      = None
        self.vectors          = None
        self.previous_options = {}
        self.block_settings   = {}

        # advanced visualization parameters,
        # including only arrow_resolution & arrow_scale
        self.adv_vis_settings = {
            'cylinder_resolution': 20,
            'cylinder_sides':      4,
            'aspect_ratio':        8,
            'draw_director':       False,
            'draw_layer_planes':   False,
            'draw_bounding_box':   False,
            'colormap_min':        0.0,
            'colormap_max':        1.0,
            'arrow_resolution':    20,
            'arrow_scale':         1.0
        }

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Point Generation and Visualization')
        main_layout    = QHBoxLayout()
        options_layout = QVBoxLayout()

        spacing_grid    = QGridLayout()
        size_grid       = QGridLayout()
        randomness_grid = QGridLayout()

        options_layout.addWidget(QLabel('Spacing:'))
        options_layout.addLayout(spacing_grid)
        options_layout.addWidget(QLabel('Size:'))
        options_layout.addLayout(size_grid)
        options_layout.addWidget(QLabel('Randomness:'))
        options_layout.addLayout(randomness_grid)

        # spacing X/Y/Z
        for i, axis in enumerate(('x','y','z')):
            inp = QLineEdit(self); inp.setFixedWidth(30)
            inp.setText(('0.6','0.6','1.2')[i])
            setattr(self, f'spacing_{axis}_input', inp)
        spacing_grid.addWidget(QLabel('X:'),0,0); spacing_grid.addWidget(self.spacing_x_input,0,1)
        spacing_grid.addWidget(QLabel('Y:'),0,2); spacing_grid.addWidget(self.spacing_y_input,0,3)
        spacing_grid.addWidget(QLabel('Z:'),0,4); spacing_grid.addWidget(self.spacing_z_input,0,5)

        # size X/Y/Z
        for i, axis in enumerate(('x','y','z')):
            inp = QLineEdit(self); inp.setFixedWidth(30)
            inp.setText(('5.0','5.0','5.0')[i])
            setattr(self, f'size_{axis}_input', inp)
        size_grid.addWidget(QLabel('X:'),0,0); size_grid.addWidget(self.size_x_input,0,1)
        size_grid.addWidget(QLabel('Y:'),0,2); size_grid.addWidget(self.size_y_input,0,3)
        size_grid.addWidget(QLabel('Z:'),0,4); size_grid.addWidget(self.size_z_input,0,5)

        # randomness X/Y/Z
        for i, axis in enumerate(('x','y','z')):
            inp = QLineEdit(self); inp.setFixedWidth(30)
            inp.setText(('0.1','0.1','0.1')[i])
            setattr(self, f'randomness_{axis}_input', inp)
        randomness_grid.addWidget(QLabel('X:'),0,0); randomness_grid.addWidget(self.randomness_x_input,0,1)
        randomness_grid.addWidget(QLabel('Y:'),0,2); randomness_grid.addWidget(self.randomness_y_input,0,3)
        randomness_grid.addWidget(QLabel('Z:'),0,4); randomness_grid.addWidget(self.randomness_z_input,0,5)

        # hexatic order & P2
        self.hexatic_checkbox = QCheckBox('In-plane hexatic order?', self)
        self.P2_label         = QLabel('P2 (Nematic Order):')
        self.P2_input         = QLineEdit(self); self.P2_input.setText('1.0')
        options_layout.addWidget(self.hexatic_checkbox)
        options_layout.addWidget(self.P2_label)
        options_layout.addWidget(self.P2_input)

        # tilt angles X/Y
        for axis in ('x','y'):
            lbl = QLabel(f'Tilt Angle {axis.upper()}:')
            inp = QLineEdit(self); inp.setText('0')
            setattr(self, f'tilt_angle_{axis}_label', lbl)
            setattr(self, f'tilt_angle_{axis}_input', inp)
            options_layout.addWidget(lbl)
            options_layout.addWidget(inp)

        # splay angles / periods / polarity
        for axis in ('x','y'):
            lbl_ang = QLabel(f'Splay Angle {axis.upper()}:')
            inp_ang = QLineEdit(self); inp_ang.setText('0')
            lbl_per = QLabel(f'Splay Period {axis.upper()}:')
            inp_per = QLineEdit(self); inp_per.setText('10')
            chk_pos = QCheckBox(f'Positive Splay {axis.upper()}', self)
            setattr(self, f'splay_angle_{axis}_label', lbl_ang)
            setattr(self, f'splay_angle_{axis}_input', inp_ang)
            setattr(self, f'splay_period_{axis}_label', lbl_per)
            setattr(self, f'splay_period_{axis}_input', inp_per)
            setattr(self, f'splay_positive_{axis}_checkbox', chk_pos)
            options_layout.addWidget(lbl_ang)
            options_layout.addWidget(inp_ang)
            options_layout.addWidget(lbl_per)
            options_layout.addWidget(inp_per)
            options_layout.addWidget(chk_pos)

        # polar toggle
        self.polar_checkbox = QCheckBox('Polar?', self)
        options_layout.addWidget(self.polar_checkbox)

        # colormap selector
        self.cmap_label    = QLabel('Color Map:')
        self.cmap_dropdown = QComboBox(self)
        valid_cmaps = set(dir(cm))
        for name in lut_mode_list():
            if name in valid_cmaps:
                pix = self.create_colormap_preview(name)
                self.cmap_dropdown.addItem(QIcon(pix), name)
        options_layout.addWidget(self.cmap_label)
        options_layout.addWidget(self.cmap_dropdown)

        # draw style + boomerang params
        self.draw_style_label    = QLabel('Draw Style:')
        self.draw_style_dropdown = QComboBox(self)
        self.draw_style_dropdown.addItems(['quiver','fast cylinder', 'cylinder','boomerang'])
        options_layout.addWidget(self.draw_style_label)
        options_layout.addWidget(self.draw_style_dropdown)

        # biaxiality
        self.B_label = QLabel('Biaxiality (<B>):')
        self.B_input = QLineEdit(self); self.B_input.setText('0.0')
        options_layout.addWidget(self.B_label)
        options_layout.addWidget(self.B_input)

        self.boomerang_angle_label = QLabel('Boomerang Angle (°):')
        self.boomerang_angle_input = QLineEdit(self); self.boomerang_angle_input.setText('60')
        options_layout.addWidget(self.boomerang_angle_label)
        options_layout.addWidget(self.boomerang_angle_input)

        # default params dropdown
        self.default_params_label    = QLabel('Default Parameters:')
        self.default_params_dropdown = QComboBox(self)
        self.default_params_dropdown.addItems(self.default_params.keys())
        self.default_params_dropdown.currentIndexChanged.connect(self.load_default_params)
        options_layout.addWidget(self.default_params_label)
        options_layout.addWidget(self.default_params_dropdown)

        # choose how we colour (color arghhh)
        self.color_by_label    = QLabel('Colour By:')
        self.color_by_dropdown = QComboBox(self)
        self.color_by_dropdown.addItems([
            'cmap','cmap (x coord)','cmap (y coord)','cmap (z coord)',
            'cmap (x rot)','cmap (y rot)','cmap (z rot)',
            'P1','P2','angle','rgb'
        ])
        self.color_by_dropdown.currentIndexChanged.connect(self.handle_color_by_change)
        self.color_picker_button = QPushButton('Choose Color', self)
        self.color_picker_button.clicked.connect(self.pick_color)
        self.color_picker_button.setVisible(False)
        self.color_value = (1,0,0)
        options_layout.addWidget(self.color_by_label)
        options_layout.addWidget(self.color_by_dropdown)
        options_layout.addWidget(self.color_picker_button)

        # action buttons
        self.generate_button      = QPushButton('Generate & Draw', self)
        self.generate_button.clicked.connect(self.generate_image)
        self.update_button        = QPushButton('Update Image', self)
        self.update_button.clicked.connect(self.update_image)
        self.block_button         = QPushButton('Block Settings', self)
        self.block_button.clicked.connect(self.show_block_settings)
        self.draw_settings_button = QPushButton('Draw Settings', self)
        self.draw_settings_button.clicked.connect(self.show_draw_settings)

        options_layout.addWidget(self.generate_button)
        options_layout.addWidget(self.update_button)
        options_layout.addWidget(self.block_button)
        options_layout.addWidget(self.draw_settings_button)

        # set the mayavi canvas
        self.mayavi_widget = MayaviQWidget(self)
        self.mayavi_widget.visualization.scene.background = (1,1,1)
        self.mayavi_widget.setMinimumSize(800,600)

        main_layout.addLayout(options_layout, 1)
        main_layout.addWidget(self.mayavi_widget, 2)
        self.setLayout(main_layout)

    def create_colormap_preview(self, name, width=100, height=20):
        grad = np.linspace(0,1,256).reshape(1,-1)
        grad = np.vstack((grad,grad))
        fig,ax = plt.subplots(figsize=(width/100,height/100),dpi=100)
        ax.imshow(grad,aspect='auto',cmap=plt.get_cmap(name))
        ax.axis('off')
        fig.subplots_adjust(0,0,1,1)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(),dtype='uint8').reshape(height,width,3)
        plt.close(fig)
        return QPixmap.fromImage(QImage(data.data,width,height,QImage.Format_RGB888))

    def handle_color_by_change(self):
        self.color_picker_button.setVisible(self.color_by_dropdown.currentText()=='rgb')

    def pick_color(self):
        c = QColorDialog.getColor()
        if c.isValid():
            self.color_value = (c.red()/255., c.green()/255., c.blue()/255.)

    def load_default_params(self):
        sel = self.default_params_dropdown.currentText()
        if sel not in self.default_params: return
        p = self.default_params[sel]
        # spacing
        self.spacing_x_input.setText(p['spacing_x'])
        self.spacing_y_input.setText(p['spacing_y'])
        self.spacing_z_input.setText(p['spacing_z'])
        # size
        self.size_x_input.setText(p['size_x'])
        self.size_y_input.setText(p['size_y'])
        self.size_z_input.setText(p['size_z'])
        # randomness
        self.randomness_x_input.setText(p['randomness_x'])
        self.randomness_y_input.setText(p['randomness_y'])
        self.randomness_z_input.setText(p['randomness_z'])
        # hexatic / P2 / polar
        self.hexatic_checkbox.setChecked(p.get('hexatic',False))
        self.P2_input.setText(p['P2'])
        self.polar_checkbox.setChecked(p.get('polar',False))
        # tilt
        self.tilt_angle_x_input.setText(p.get('tilt_angle_x','0'))
        self.tilt_angle_y_input.setText(p.get('tilt_angle_y','0'))
        # blocks
        blocks = p.get('blocks',{})
        self.block_settings = {
            'block_distance_x': float(blocks.get('block_distance_x',0)),
            'block_rotation_x': float(blocks.get('block_rotation_x',0)),
            'block_distance_y': float(blocks.get('block_distance_y',0)),
            'block_rotation_y': float(blocks.get('block_rotation_y',0)),
            'block_distance_z': float(blocks.get('block_distance_z',0)),
            'block_rotation_z': float(blocks.get('block_rotation_z',0)),
            'draw_x_block_plane': blocks.get('draw_x_block_plane',False),
            'draw_y_block_plane': blocks.get('draw_y_block_plane',False),
            'draw_z_block_plane': blocks.get('draw_z_block_plane',False)
        }

    def show_block_settings(self):
        dlg = BlockSettingsDialog()
        if dlg.exec_():
            self.block_settings = dlg.get_block_settings()

    def show_draw_settings(self):
        dlg = AdvancedVisualizationSettings(self.adv_vis_settings)
        if dlg.exec_():
            self.adv_vis_settings = dlg.get_settings()

    def generate_image(self):
        self.coordinates = None
        self.update_image()

    def update_image(self):
        spacing   = [float(self.spacing_x_input.text()),
                     float(self.spacing_y_input.text()),
                     float(self.spacing_z_input.text())]
        size      = [float(self.size_x_input.text()),
                     float(self.size_y_input.text()),
                     float(self.size_z_input.text())]
        randomness= [float(self.randomness_x_input.text()),
                     float(self.randomness_y_input.text()),
                     float(self.randomness_z_input.text())]
        P2        = float(self.P2_input.text())
        tilt_x    = float(self.tilt_angle_x_input.text())
        tilt_y    = float(self.tilt_angle_y_input.text())
        splay_x   = float(self.splay_angle_x_input.text())
        splay_px  = float(self.splay_period_x_input.text())
        splay_y   = float(self.splay_angle_y_input.text())
        splay_py  = float(self.splay_period_y_input.text())
        pos_sx    = self.splay_positive_x_checkbox.isChecked()
        pos_sy    = self.splay_positive_y_checkbox.isChecked()
        polar     = self.polar_checkbox.isChecked()

        cmap          = self.cmap_dropdown.currentText()
        draw_style    = self.draw_style_dropdown.currentText()
        color_by      = self.color_by_dropdown.currentText()

        B             = float(self.B_input.text())
        B             = max(0.0, min(1.0, B))
        boomerang_ang = float(self.boomerang_angle_input.text())

        opts = {
            'spacing': spacing, 'size': size, 'randomness': randomness,
            'P2': P2,
            'tilt_x': tilt_x, 'tilt_y': tilt_y,
            'splay_x': splay_x, 'splay_px': splay_px,
            'splay_y': splay_y, 'splay_py': splay_py,
            'pos_sx': pos_sx, 'pos_sy': pos_sy,
            'polar': polar,
            'block_settings': self.block_settings,
            'B': B, 'boomerang_angle': boomerang_ang,
            'arrow_resolution': self.adv_vis_settings['arrow_resolution'],
            'arrow_scale':      self.adv_vis_settings['arrow_scale']
        }

        if self.coordinates is None or opts != self.previous_options:
            coords = generate_regular_points(spacing, size)
            if self.hexatic_checkbox.isChecked():
                coords = hexatic_offset(coords, plane_offset=True)
            coords = add_randomness(coords, *randomness)
            vecs   = define_vectors(coords, 1, P2)
            vecs, coords = self.apply_splay(
                coords, vecs,
                splay_x, splay_y,
                pos_sx, pos_sy,
                splay_px, splay_py
            )
            coords, vecs = add_tilt(coords, vecs, tilt_x, tilt_y)
            self.coordinates, self.vectors = coords, vecs
            self.apply_block_rotations()
        else:
            coords, vecs = self.coordinates, self.vectors

        if not polar:
            idx = np.random.choice(len(vecs), len(vecs)//2, replace=False)
            coords[idx] += vecs[idx]
            vecs[idx]   = -vecs[idx]

        P1, P1_arr, P2_act, P2_arr, avg_ang, angles = calculate_average_angle(vecs)
        props = np.vstack((P1_arr, P2_arr, angles))

        self.mayavi_widget.visualize_points(
            coords, vecs,
            draw_style,
            cmap,
            self.adv_vis_settings['aspect_ratio'],
            self.adv_vis_settings['draw_bounding_box'],
            color_by, self.color_value,
            props,
            self.adv_vis_settings['colormap_min'],
            self.adv_vis_settings['colormap_max'],
            tube_length=self.adv_vis_settings['cylinder_resolution'],
            tube_sides =self.adv_vis_settings['cylinder_sides'],
            biaxiality=B,
            boomerang_angle=boomerang_ang,
            arrow_resolution=self.adv_vis_settings['arrow_resolution'],
            arrow_scale     =self.adv_vis_settings['arrow_scale']
        )

        if self.adv_vis_settings['draw_layer_planes']:
            self.draw_layer_planes(spacing[2])
        if self.adv_vis_settings['draw_director']:
            dirn = compute_director(vecs)
            self.mayavi_widget.draw_director(dirn, coords)
        self.draw_block_planes()

        self.previous_options = opts

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
                self.coordinates[i] = point - np.array([0, 0, np.cos(np.radians(angle))/2])
                self.vectors[i] = self.rotate_vector(vector, angle, axis='x')

        if block_distance_y > 0:
            for i, (point, vector) in enumerate(zip(self.coordinates, self.vectors)):
                step = int(point[1] // block_distance_y)
                angle = step * block_rotation_y
                self.coordinates[i] = point - np.array([0, 0, np.cos(np.radians(angle))/2])
                self.vectors[i] = self.rotate_vector(vector, angle, axis='y')

        if block_distance_z > 0:
            for i, (point, vector) in enumerate(zip(self.coordinates, self.vectors)):
                step = int(point[2] // block_distance_z)
                angle = step * block_rotation_z
                self.vectors[i] = self.rotate_vector(vector, angle, axis='z')

    def rotate_vector(self, vector, angle, axis='x'):
        angle_rad = np.radians(angle)
        if axis == 'x':
            R = np.array([[1, 0, 0],
                          [0, np.cos(angle_rad), -np.sin(angle_rad)],
                          [0, np.sin(angle_rad),  np.cos(angle_rad)]])
        elif axis == 'y':
            R = np.array([[ np.cos(angle_rad), 0, np.sin(angle_rad)],
                          [0, 1, 0],
                          [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
        else:
            R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                          [np.sin(angle_rad),  np.cos(angle_rad), 0],
                          [0, 0, 1]])
        return R.dot(vector)

    def draw_layer_planes(self, spacing_z):
        block_distance_z = self.block_settings.get('block_distance_z', 0)
        if block_distance_z == 0:
            self.draw_planes_in_range(0, np.max(self.coordinates[:, 2]), spacing_z, 0)
        else:
            max_z = np.max(self.coordinates[:, 2])
            for z_start in np.arange(0, max_z, block_distance_z):
                z_end = min(z_start + block_distance_z, max_z)
                # no additional rotation
                self.draw_planes_in_range(z_start, z_end, spacing_z, 0)

    def draw_planes_in_range(self, z_start, z_end, spacing_z, block_rotation_z):
        for z in np.arange(z_start, z_end, spacing_z):
            self.draw_plane(z, 'z', block_rotation_z)

    def draw_plane(self, position, axis, tilt_angle=0):
        if axis == 'x':
            y = np.linspace(np.min(self.coordinates[:, 1]), np.max(self.coordinates[:, 1]), 100)
            z = np.linspace(np.min(self.coordinates[:, 2]), np.max(self.coordinates[:, 2]), 100)
            y, z = np.meshgrid(y, z)
            x = np.full_like(y, position)
            self.mayavi_widget.visualization.scene.mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=0.3)
        elif axis == 'y':
            x = np.linspace(np.min(self.coordinates[:, 0]), np.max(self.coordinates[:, 0]), 100)
            z = np.linspace(np.min(self.coordinates[:, 2]), np.max(self.coordinates[:, 2]), 100)
            x, z = np.meshgrid(x, z)
            y = np.full_like(x, position)
            self.mayavi_widget.visualization.scene.mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=0.3)
        else:  # 'z'
            x = np.linspace(np.min(self.coordinates[:, 0]), np.max(self.coordinates[:, 0]), 100)
            y = np.linspace(np.min(self.coordinates[:, 1]), np.max(self.coordinates[:, 1]), 100)
            x, y = np.meshgrid(x, y)
            z = np.full_like(x, position)
            if tilt_angle != 0:
                x, y, z = self.apply_tilt(x, y, z, tilt_angle)
            self.mayavi_widget.visualization.scene.mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=0.3)

    def apply_tilt(self, x, y, z, tilt_angle):
        tilt_rad = np.radians(tilt_angle)
        x_new = x * np.cos(tilt_rad) - z * np.sin(tilt_rad)
        z_new = x * np.sin(tilt_rad) + z * np.cos(tilt_rad)
        return x_new, y, z_new

    def apply_splay(self, coordinates, vectors,
                    splay_angle_x, splay_angle_y,
                    positive_splay_x, positive_splay_y,
                    splay_period_x, splay_period_y):
        for i, (coord, vec) in enumerate(zip(coordinates, vectors)):
            # X-splay
            if splay_angle_x != 0 and splay_period_x > 0:
                offset_x = (coord[0] % splay_period_x) / splay_period_x
                if positive_splay_x:
                    angle = (-0.5 + offset_x) * 2 * splay_angle_x
                    coordinates[i][0] -= np.sin(np.radians(angle))
                else:
                    angle = (0.5 - offset_x) * 2 * splay_angle_x
                    coordinates[i][0] += np.sin(np.radians(angle))
                vectors[i] = self.rotate_vector(vec, angle, 'y')

            # Y-splay
            if splay_angle_y != 0 and splay_period_y > 0:
                offset_y = (coord[1] % splay_period_y) / splay_period_y
                if positive_splay_y:
                    angle = (-0.5 + offset_y) * 2 * splay_angle_y
                    coordinates[i][1] -= np.sin(np.radians(angle))
                else:
                    angle = (0.5 - offset_y) * 2 * splay_angle_y
                    coordinates[i][1] += np.sin(np.radians(angle))
                vectors[i] = self.rotate_vector(vec, angle, 'x')

        return vectors, coordinates

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex  = MainApp()
    ex.show()
    sys.exit(app.exec_())