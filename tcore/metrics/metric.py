import open3d as o3d
import torch
import numpy as np

from tcore.metrics import MESHTYPE, TETRATYPE, PCDTYPE


class Metrics3D():

    def prediction_is_empty(self, geom):

        if isinstance(geom, o3d.geometry.Geometry):
            geom_type = geom.get_geometry_type().value
            if geom_type == MESHTYPE:
                geom.remove_duplicated_vertices()
                geom.remove_duplicated_triangles()
                geom.remove_degenerate_triangles()
                empty_v = self.is_empty(len(geom.vertices))
                empty_t = self.is_empty(len(geom.triangles))
                empty = empty_t or empty_v
            elif geom_type == TETRATYPE:
                geom.remove_duplicated_vertices()
                geom.remove_duplicated_tetras()
                geom.remove_degenerate_tetras()
                empty_v = self.is_empty(len(geom.vertices))
                empty_t = self.is_empty(len(geom.tetras))
                empty = empty_t or empty_v
            elif geom_type == PCDTYPE:
                empty = self.is_empty(len(geom.points))
            else:
                assert False, '{} geometry not supported'.format(geom.get_geometry_type())
        elif isinstance(geom, np.ndarray):
            empty = self.is_empty(len(geom[:, :3]))
        elif isinstance(geom, torch.Tensor):
            empty = self.is_empty(len(geom[:, :3]))
        else:
            assert False, '{} type not supported'.format(type(geom))

        return empty

    @staticmethod
    def convert_to_pcd(geom):

        if isinstance(geom, o3d.geometry.Geometry):
            geom_type = geom.get_geometry_type().value
            if geom_type == MESHTYPE or geom_type == TETRATYPE:
                geom_pcd = geom.sample_points_uniformly(1000000)
            elif geom_type == PCDTYPE:
                geom_pcd = geom
            else:
                assert False, '{} geometry not supported'.format(geom.get_geometry_type())
        elif isinstance(geom, np.ndarray):
            geom_pcd = o3d.geometry.PointCloud()
            geom_pcd.points = o3d.utility.Vector3dVector(geom[:, :3])
        elif isinstance(geom, torch.Tensor):
            geom = geom.detach().cpu().numpy()
            geom_pcd = o3d.geometry.PointCloud()
            geom_pcd.points = o3d.utility.Vector3dVector(geom[:, :3])
        else:
            assert False, '{} type not supported'.format(type(geom))

        return geom_pcd

    @staticmethod
    def is_empty(length):
        empty = True
        if length:
            empty = False
        return empty