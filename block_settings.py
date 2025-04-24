from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QPushButton, QCheckBox

'''
This controls the block settings popup.
'''

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

        self.draw_x_block_plane_checkbox = QCheckBox('Draw X Block Plane?', self)
        self.draw_y_block_plane_checkbox = QCheckBox('Draw Y Block Plane?', self)
        self.draw_z_block_plane_checkbox = QCheckBox('Draw Z Block Plane?', self)

        layout.addRow('Block Distance X:', self.block_distance_x_input)
        layout.addRow('Block Rotation X:', self.block_rotation_x_input)
        layout.addRow('Block Distance Y:', self.block_distance_y_input)
        layout.addRow('Block Rotation Y:', self.block_rotation_y_input)
        layout.addRow('Block Distance Z:', self.block_distance_z_input)
        layout.addRow('Block Rotation Z:', self.block_rotation_z_input)
        layout.addRow(self.draw_x_block_plane_checkbox)
        layout.addRow(self.draw_y_block_plane_checkbox)
        layout.addRow(self.draw_z_block_plane_checkbox)

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
            'block_rotation_z': float(self.block_rotation_z_input.text()),
            'draw_x_block_plane': self.draw_x_block_plane_checkbox.isChecked(),
            'draw_y_block_plane': self.draw_y_block_plane_checkbox.isChecked(),
            'draw_z_block_plane': self.draw_z_block_plane_checkbox.isChecked()
        }
