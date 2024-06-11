from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QFormLayout, QCheckBox, QPushButton

class AdvancedVisualizationSettings(QDialog):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Advanced Visualization Settings')
        layout = QFormLayout()

        self.cylinder_resolution_input = QLineEdit(self)
        self.cylinder_resolution_input.setText(str(self.settings['cylinder_resolution']))
        self.cylinder_sides_input = QLineEdit(self)
        self.cylinder_sides_input.setText(str(self.settings['cylinder_sides']))
        self.aspect_ratio_input = QLineEdit(self)
        self.aspect_ratio_input.setText(str(self.settings['aspect_ratio']))

        self.draw_director_checkbox = QCheckBox('Draw Director?', self)
        self.draw_director_checkbox.setChecked(self.settings['draw_director'])
        self.draw_layer_planes_checkbox = QCheckBox('Draw Layer Planes?', self)
        self.draw_layer_planes_checkbox.setChecked(self.settings['draw_layer_planes'])
        self.draw_bounding_box_checkbox = QCheckBox('Draw Bounding Box?', self)
        self.draw_bounding_box_checkbox.setChecked(self.settings['draw_bounding_box'])

        self.colormap_min_input = QLineEdit(self)
        self.colormap_min_input.setText(str(self.settings.get('colormap_min', 0.0)))
        self.colormap_max_input = QLineEdit(self)
        self.colormap_max_input.setText(str(self.settings.get('colormap_max', 1.0)))

        layout.addRow('Cylinder Resolution:', self.cylinder_resolution_input)
        layout.addRow('Cylinder Sides:', self.cylinder_sides_input)
        layout.addRow('Aspect Ratio:', self.aspect_ratio_input)
        layout.addRow(self.draw_director_checkbox)
        layout.addRow(self.draw_layer_planes_checkbox)
        layout.addRow(self.draw_bounding_box_checkbox)
        layout.addRow('Colormap Min:', self.colormap_min_input)
        layout.addRow('Colormap Max:', self.colormap_max_input)

        save_button = QPushButton('Save', self)
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)

        self.setLayout(layout)

    def save_settings(self):
        self.settings['cylinder_resolution'] = int(self.cylinder_resolution_input.text())
        self.settings['cylinder_sides'] = int(self.cylinder_sides_input.text())
        self.settings['aspect_ratio'] = float(self.aspect_ratio_input.text())
        self.settings['draw_director'] = self.draw_director_checkbox.isChecked()
        self.settings['draw_layer_planes'] = self.draw_layer_planes_checkbox.isChecked()
        self.settings['draw_bounding_box'] = self.draw_bounding_box_checkbox.isChecked()
        self.settings['colormap_min'] = float(self.colormap_min_input.text())
        self.settings['colormap_max'] = float(self.colormap_max_input.text())
        self.accept()

    def get_settings(self):
        return self.settings