import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from mayavi import mlab

'''
This is the actual visualisation engine that tranlates our generated coordinates into something
that mayavi can actually draw. 

Too much of the logic is in here. It should just work as a wrapper for mayavi (I Think).
'''

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

    def visualize_points(
        self, points, vectors=None,
        draw_style='quiver', cmap='viridis',
        aspect_ratio=8, draw_box=False,
        color_by='cmap', color_value=(1,0,0),
        property_array=None, colormap_min=0.0, colormap_max=1.0,
        tube_length=20, tube_sides=4,
        biaxiality=0.0, boomerang_angle=60,
        arrow_resolution=20, arrow_scale=1.0
    ):
        """
        Render points/vectors in various styles.
        - arrow_resolution: shaft & tip resolution of quiver glyphs
        - arrow_scale:      overall scale_factor of quiver glyphs
        """
        self.scene.mlab.clf()
        self.cylinders = []

        # 1) no vectors, only draw points (not used)
        if vectors is None:
            self.scene.mlab.points3d(
                points[:,0], points[:,1], points[:,2],
                scale_factor=0.1, colormap=cmap
            )
            self.scene.render()
            return

        # 2) scalar mapping (unless rgb)
        if color_by == 'rgb':
            scalars = None
        else:
            scalars = self.get_scalars(
                vectors, color_by, color_value,
                property_array, points,
                colormap_min, colormap_max
            )

        # 3) use mlab.quiver3D to draw 3D Arrows (this is fast and is now the default)
        if draw_style in ('quiver','quiver-3D'):
            q = self.scene.mlab.quiver3d(
                points[:,0],points[:,1],points[:,2],
                vectors[:,0],vectors[:,1],vectors[:,2],
                scalars=scalars, colormap=cmap,
                mode='arrow', scale_factor=arrow_scale
            )

            q.glyph.glyph_source.glyph_source.shaft_resolution = arrow_resolution
            q.glyph.glyph_source.glyph_source.tip_resolution   = arrow_resolution

            if scalars is None:
                q.actor.mapper.scalar_visibility = False
                q.actor.property.color = color_value
            else:
                q.glyph.color_mode = 'color_by_scalar'
        
        # 4) fast cylinder style. This uses quiver3D (so is fast) but is limited in terms of colouring.
        
        elif draw_style == 'fast cylinder':
            
            glyphs = []
            cyl_radius = 1.0 / float(aspect_ratio) 
            
            glyph_radius_correction_factor = 2 # this is an empirical, hard-coded (uh oh) factor to make the "caps" the same radius as the cylinders.
          
            starts = points
            ends = points + vectors
            
            q1 = self.scene.mlab.quiver3d(
                points[:,0],points[:,1],points[:,2],
                vectors[:,0],vectors[:,1],vectors[:,2],
                scalars=scalars, colormap=cmap,
                mode='cylinder', 
                resolution=arrow_resolution, 
                scale_factor=arrow_scale,
            )
            q1.glyph.glyph_source.glyph_source.radius = cyl_radius
            glyphs.append(q1)
            
            q2 = self.scene.mlab.points3d(
                starts[:,0],starts[:,1],starts[:,2],
                *([scalars] if scalars is not None else []),
                colormap=cmap if scalars is not None else None,
                scale_factor=cyl_radius * glyph_radius_correction_factor,
                mode='sphere',
                resolution=arrow_resolution, 
                scale_mode = 'none'
            )
            glyphs.append(q2)
            
            q3 = self.scene.mlab.points3d(
                ends[:,0], ends[:,1], ends[:,2],
                *([scalars] if scalars is not None else []),
                colormap=cmap if scalars is not None else None,
                scale_factor=cyl_radius * glyph_radius_correction_factor,
                mode='sphere',
                resolution=arrow_resolution, 
                scale_mode = 'none'
            )
            glyphs.append(q3)
            
            for q in glyphs:
                if scalars is None:
                    q.actor.mapper.scalar_visibility = False
                    q.actor.property.color = color_value
                else:
                    q.glyph.color_mode = 'color_by_scalar'
                    
        # 5) cylinder style - this is slow, but it allows colour gradients to be placed along the cylinder
        # length which is desirable for some things.
        elif draw_style == 'cylinder':
            radius = 1.0 / float(aspect_ratio)
            for i,(pt,vec) in enumerate(zip(points,vectors)):
                if not self.should_draw_cylinder(i, points, radius):
                    continue
                scalar = None if scalars is None else scalars[i]
                if scalars is None:
                    col = color_value
                elif color_by!='cmap':
                    col = self.normalize_color(scalar)
                else:
                    col = None
                tube = self.draw_cylinder(
                    pt, vec, radius, col, cmap, color_by,
                    tube_length, tube_sides
                )
                self.cylinders.append(tube)

        # 6) Boomerang: two quivers both seated at the midpoint,
        elif draw_style == 'boomerang':
            Bq = max(0.0, min(1.0, biaxiality))
            theta = np.radians(boomerang_angle)

            mids = []
            dirs1 = []
            dirs2 = []
                        
            for pt, vec in zip(points, vectors):
                length = np.linalg.norm(vec)
                dirn   = vec / length

                up   = np.array([0,0,1])
                perp = np.cross(dirn, up)
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.cross(dirn, np.array([0,1,0]))
                perp /= np.linalg.norm(perp)

                phi  = np.random.rand() * 2*np.pi * (1 - Bq)
                perp = (perp*np.cos(phi)
                        + np.cross(dirn, perp)*np.sin(phi)
                        + dirn*np.dot(dirn,perp)*(1 - np.cos(phi)))

                half   = length/2
                offset = half * np.tan(theta/2)
                mid_pt = pt + dirn*half + perp*offset

                mids.append(mid_pt)
                dirs1.append(pt - mid_pt)       # arrow toward start
                dirs2.append((pt+vec) - mid_pt) # arrow toward end

            mids  = np.array(mids)
            dirs1 = np.array(dirs1)
            dirs2 = np.array(dirs2)
            
            starts = mids + dirs1
            ends   = mids + dirs2
            
            cyl_radius = 1.0 / float(aspect_ratio) 
            
            glyphs = []
            
            glyph_radius_correction_factor = 1.15 # this is an empirical, hard-coded (uh oh) factor to make the "caps" the same radius as the cylinders.
            # half 1
            q1 = self.scene.mlab.quiver3d(
                mids[:,0], mids[:,1], mids[:,2],
                dirs1[:,0],dirs1[:,1],dirs1[:,2],
                scalars=scalars, colormap=cmap,
                resolution=arrow_resolution,
                mode='cylinder', scale_factor=arrow_scale
            )
            q1.glyph.glyph_source.glyph_source.radius = cyl_radius
            glyphs.append(q1)

            # half 2
            q2 = self.scene.mlab.quiver3d(
                mids[:,0], mids[:,1], mids[:,2],
                dirs2[:,0],dirs2[:,1],dirs2[:,2],
                scalars=scalars, colormap=cmap,
                resolution=arrow_resolution,
                mode='cylinder', scale_factor=arrow_scale
            )
            q2.glyph.glyph_source.glyph_source.radius = cyl_radius
            glyphs.append(q2)
            
            # put a sphere in the middle of the two points to join them up visualy.
            q3 = self.scene.mlab.points3d(
                mids[:,0], mids[:,1], mids[:,2],
                *([scalars] if scalars is not None else []),
                colormap=cmap if scalars is not None else None,
                scale_factor=cyl_radius * glyph_radius_correction_factor,
                mode='sphere',
                resolution=arrow_resolution, 
                scale_mode = 'none'
            )
            glyphs.append(q3)
                
            # sphere at each start-point
            q4 = self.scene.mlab.points3d(
                starts[:,0], starts[:,1], starts[:,2],
                *([scalars] if scalars is not None else []),
                colormap=cmap if scalars is not None else None,
                scale_factor=cyl_radius * glyph_radius_correction_factor,
                mode='sphere',
                resolution=arrow_resolution, 
                scale_mode = 'none'
            )
            glyphs.append(q4
            )

            # sphere at each end-point
            q5 = self.scene.mlab.points3d(
                ends[:,0], ends[:,1], ends[:,2],
                *([scalars] if scalars is not None else []),
                colormap=cmap if scalars is not None else None,
                scale_factor=cyl_radius * glyph_radius_correction_factor,
                mode='sphere',
                resolution=arrow_resolution, 
                scale_mode = 'none'
            )
            glyphs.append(q5)
            for q in glyphs:
                #q.glyph.scale_mode = 'data_scaling_off'
                if scalars is None:
                    q.actor.mapper.scalar_visibility = False
                    q.actor.property.color = color_value
                else:
                    q.glyph.color_mode = 'color_by_scalar'
            
        # 7) Fallback: default quiver
        else:
            q = self.scene.mlab.quiver3d(
                points[:,0],points[:,1],points[:,2],
                vectors[:,0],vectors[:,1],vectors[:,2],
                scalars=scalars, colormap=cmap,
                mode='arrow', scale_factor=arrow_scale
            )
            q.glyph.glyph_source.glyph_source.shaft_resolution = arrow_resolution
            q.glyph.glyph_source.glyph_source.tip_resolution   = arrow_resolution
            q.glyph.color_mode = 'color_by_scalar'

        # Colorbar if needed
        if color_by in ('P1','P2','angle'):
            cb = self.scene.mlab.colorbar(title=color_by, orientation='vertical')
            if cb:
                sb = cb.scalar_bar
                sb.label_text_property.color = (0,0,0)
                sb.title_text_property.color = (0,0,0)

        # Bounding box?
        if draw_box:
            self.draw_bounding_box(points, vectors)

        self.scene.render()

    def should_draw_cylinder(self, index, points, radius):
        for i, p in enumerate(points):
            if i!=index and np.linalg.norm(points[index]-p)<radius:
                return False
        return True

    def draw_cylinder(self, point, vector, radius, color, cmap, color_by, tube_length=20, tube_sides=4):
        x = np.linspace(point[0], point[0]+vector[0], tube_length)
        y = np.linspace(point[1], point[1]+vector[1], tube_length)
        z = np.linspace(point[2], point[2]+vector[2], tube_length)
        if color_by=='cmap':
            tube = mlab.plot3d(x,y,z, np.linspace(0,1,tube_length),
                               tube_radius=radius, tube_sides=tube_sides, colormap=cmap)
            tube.module_manager.scalar_lut_manager.lut_mode = cmap
            tube.module_manager.scalar_lut_manager.data_range = (0,1)
        else:
            if isinstance(color,tuple):
                tube = mlab.plot3d(x,y,z, np.linspace(0,1,tube_length),
                                   tube_radius=radius, tube_sides=tube_sides)
                lut = tube.module_manager.scalar_lut_manager.lut.table.to_array()
                lut[:,:3] = np.array(color)*255
                tube.module_manager.scalar_lut_manager.lut.table = lut
            else:
                tube = mlab.plot3d(x,y,z, np.full(tube_length, color),
                                   tube_radius=radius, tube_sides=tube_sides, colormap=cmap)
                tube.module_manager.scalar_lut_manager.lut_mode = cmap
        return tube

    def update_cylinder_colors(self, scalars, color_by, color_value, property_array, tube_length=20):
        for i, cyl in enumerate(self.cylinders):
            if color_by == 'cmap':
                cyl.mlab_source.scalars = np.linspace(0,1,tube_length)
            else:
                cyl.actor.property.color = self.normalize_color(scalars[i])

    def normalize_color(self, scalar):
        try:
            arr = np.array(scalar)
            if arr.size == 3:
                return tuple(np.clip(arr, 0, 1))
        except:
            pass
        v = max(0, min(1, float(scalar)))
        return (v, v, v)

    def get_scalars(self, vectors, color_by, color_value, property_array, points, colormap_min=0.0, colormap_max=1.0):
        if color_by == 'P1':
            scalars = property_array[0]
        elif color_by == 'P2':
            scalars = property_array[1]
        elif color_by == 'angle':
            scalars = np.degrees(property_array[2])
        elif color_by == 'cmap (x coord)':
            scalars = points[:,0]
        elif color_by == 'cmap (y coord)':
            scalars = points[:,1]
        elif color_by == 'cmap (z coord)':
            scalars = points[:,2]
        elif color_by == 'cmap (x rot)':
            scalars = np.arctan2(vectors[:,1],vectors[:,2])
        elif color_by == 'cmap (y rot)':
            scalars = np.arctan2(vectors[:,0],vectors[:,2])
        elif color_by == 'cmap (z rot)':
            scalars = np.arctan2(vectors[:,1],vectors[:,0])
        elif color_by == 'cmap':
            scalars = np.linspace(0,1,len(vectors))
        else:
            scalars = np.linspace(0,1,len(vectors))
        return np.interp(scalars, (scalars.min(),scalars.max()), (colormap_min,colormap_max))

    def draw_bounding_box(self, points, vectors):
        x_min = np.min(np.concatenate([points[:,0], points[:,0]+vectors[:,0]]))
        x_max = np.max(np.concatenate([points[:,0], points[:,0]+vectors[:,0]]))
        y_min = np.min(np.concatenate([points[:,1], points[:,1]+vectors[:,1]]))
        y_max = np.max(np.concatenate([points[:,1], points[:,1]+vectors[:,1]]))
        z_min = np.min(np.concatenate([points[:,2], points[:,2]+vectors[:,2]]))
        z_max = np.max(np.concatenate([points[:,2], points[:,2]+vectors[:,2]]))
        lines = [
            ((x_min,y_min,z_min),(x_max,y_min,z_min)),
            ((x_max,y_min,z_min),(x_max,y_max,z_min)),
            ((x_max,y_max,z_min),(x_min,y_max,z_min)),
            ((x_min,y_max,z_min),(x_min,y_min,z_min)),
            ((x_min,y_min,z_max),(x_max,y_min,z_max)),
            ((x_max,y_min,z_max),(x_max,y_max,z_max)),
            ((x_max,y_max,z_max),(x_min,y_max,z_max)),
            ((x_min,y_max,z_max),(x_min,y_min,z_max)),
            ((x_min,y_min,z_min),(x_min,y_min,z_max)),
            ((x_max,y_min,z_min),(x_max,y_min,z_max)),
            ((x_max,y_max,z_min),(x_max,y_max,z_max)),
            ((x_min,y_max,z_min),(x_min,y_max,z_max)),
        ]
        for (s,e) in lines:
            self.scene.mlab.plot3d([s[0],e[0]], [s[1],e[1]], [s[2],e[2]],
                                   color=(0,0,0), line_width=1.0)

    def draw_director(self, director, points):
        center = np.mean(points,axis=0)
        length = 2.5
        start  = center - director*length/2
        end    = center + director*length/2
        self.scene.mlab.plot3d([start[0],end[0]], [start[1],end[1]], [start[2],end[2]],
                               color=(1,0,0), tube_radius=0.1)

    def draw_plane(self, coordinates, position, axis, tilt_angle=0):
        if axis=='x':
            y = np.linspace(coordinates[:,1].min(),coordinates[:,1].max(),100)
            z = np.linspace(coordinates[:,2].min(),coordinates[:,2].max(),100)
            y,z=np.meshgrid(y,z)
            x=np.full_like(y,position)
            self.scene.mlab.mesh(x,y,z,color=(0.5,0.5,0.5),opacity=0.3)
        elif axis=='y':
            x = np.linspace(coordinates[:,0].min(),coordinates[:,0].max(),100)
            z = np.linspace(coordinates[:,2].min(),coordinates[:,2].max(),100)
            x,z=np.meshgrid(x,z)
            y=np.full_like(x,position)
            self.scene.mlab.mesh(x,y,z,color=(0.5,0.5,0.5),opacity=0.3)
        else:
            x = np.linspace(coordinates[:,0].min(),coordinates[:,0].max(),100)
            y = np.linspace(coordinates[:,1].min(),coordinates[:,1].max(),100)
            x,y=np.meshgrid(x,y)
            z=np.full_like(x,position)
            if tilt_angle!=0:
                x,y,z=self.apply_tilt(x,y,z,tilt_angle)
            self.scene.mlab.mesh(x,y,z,color=(0.5,0.5,0.5),opacity=0.3)

    def apply_tilt(self, x, y, z, tilt_angle):
        tr = np.radians(tilt_angle)
        x_new = x*np.cos(tr) - z*np.sin(tr)
        z_new = x*np.sin(tr) + z*np.cos(tr)
        return x_new, y, z_new

class MayaviQWidget(QWidget):
    def __init__(self, parent=None):
        super(MayaviQWidget, self).__init__(parent)
        layout = QVBoxLayout(self)
        self.visualization = Visualization()
        self.scene_widget   = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.scene_widget)
        self.setLayout(layout)

    def visualize_points(self, *args, **kwargs):
        self.visualization.visualize_points(*args,**kwargs)

    def update_cylinder_colors(self, *args, **kwargs):
        self.visualization.update_cylinder_colors(*args,**kwargs)

    def draw_director(self, *args, **kwargs):
        self.visualization.draw_director(*args,**kwargs)

    def draw_plane(self, *args, **kwargs):
        self.visualization.draw_plane(*args,**kwargs)
