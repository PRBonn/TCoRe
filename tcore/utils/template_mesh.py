import open3d as o3d
import torch
import numpy as np


class TemplateMesh():

    def __init__(self):
        self.vertices, self.faces = self.create_3D_template_mesh() 

    def get_vertices_faces(self):
        return self.vertices, self.faces

    def create_3D_template_mesh(self):
        ico = o3d.geometry.TriangleMesh.create_icosahedron(radius=0.04)
        template = ico.subdivide_loop(number_of_iterations=4)

        vertices = torch.from_numpy(np.asarray(
            template.vertices)).cuda().unsqueeze(0).float()
        faces = torch.from_numpy(np.asarray(
            template.triangles)).cuda().unsqueeze(0)

        return vertices, faces
