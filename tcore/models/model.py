import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
from pytorch3d.loss import chamfer_distance as ChamferDistanceLoss
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.structures import Meshes
from pytorch_lightning import LightningModule
from tcore.models.backbone import MinkEncoderDecoder
from tcore.models.decoder import TransformerDecoder
from tcore.utils.template_mesh import TemplateMesh
from tcore.metrics.chamfer_distance import ChamferDistance
from tcore.metrics.precision_recall import PrecisionRecall

class TCoRe(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(dict(hparams))
        self.cfg = hparams

        template = TemplateMesh()

        self.template_points, self.template_faces = template.get_vertices_faces()

        hparams.DECODER.NUM_QUERIES = self.template_points.shape[1]

        backbone = MinkEncoderDecoder(
            hparams.BACKBONE, template_points=self.template_points
        )
        self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(
            backbone)

        self.decoder = TransformerDecoder(
            hparams.DECODER,
            hparams.BACKBONE
        )

        self.freezeModules()
        self.chamfer_dist = ChamferDistanceLoss

        self.chamfer_dist_metric = ChamferDistance()
        self.precision_recall = PrecisionRecall(
            min_t=0.001, max_t=0.01, num=100)

        self.offset_scaling = self.cfg.DECODER.OFFSET_SCALING
        self.fruit = self.cfg.MODEL.FRUIT

        if self.fruit == "SWEETPEPPER":
            self.smooth = self.cfg.MODEL.SMOOTH_SW
        elif self.fruit == "STRAWBERRY":
            self.smooth = self.cfg.MODEL.SMOOTH_ST

    def freezeModules(self):
        freeze_dict = {"BACKBONE": self.backbone, "DECODER": self.decoder}
        print("Frozen modules: ", self.cfg.TRAIN.FREEZE_MODULES)
        for module in self.cfg.TRAIN.FREEZE_MODULES:
            for param in freeze_dict[module].parameters():
                param.requires_grad = False

    def forward(self, x):
        batch_size = len(x["points"])
        template_points = []
        template_faces = []
        for _ in range(batch_size):
            template = TemplateMesh()
            pts, faces = template.get_vertices_faces()

            template_points.append(pts)
            template_faces.append(faces)

        feats, coors, pad_masks = self.backbone(x)
        outputs = self.decoder(feats, coors, pad_masks, self.template_points)
        return outputs

    def training_step(self, x: dict, idx):
        losses = {}

        outputs = self.forward(x)
        losses = self.get_loss(x, outputs, losses, "loss_cd")

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                losses = self.get_loss(
                    x, aux_outputs, losses, "loss_cd_" + str(i))

        total_loss = sum(losses.values())

        loss_cd = 0
        loss_reg_laplacian = 0
        loss_reg_normals = 0
        for key in losses.keys():
            if "cd" in key:
                loss_cd += losses[key]
            if "laplacian" in key:
                loss_reg_laplacian += losses[key]
            if "normals" in key:
                loss_reg_normals += losses[key]

        self.log("train_loss", total_loss,
                 batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log("train_loss_cd", loss_cd,
                 batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log(
            "train_loss_reg_normals",
            loss_reg_normals,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
        )
        self.log(
            "train_loss_reg_laplacian",
            loss_reg_laplacian,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
        )

        torch.cuda.empty_cache()  # KEEP THIS FOR MINKOWSKI
        return total_loss

    def validation_step(self, x: dict, idx):
        outputs = self.forward(x)
        deformed_template = self.deform_template(outputs=outputs)

        for b_idx in range(len(outputs["offsets"])):
            pt = deformed_template[b_idx]
            gt = x["extra"]["gt_points"][b_idx]

            pt_mesh = o3d.geometry.TriangleMesh()
            pt_mesh.vertices = o3d.utility.Vector3dVector(pt.cpu())
            pt_mesh.triangles = o3d.utility.Vector3iVector(
                self.template_faces[0].cpu())

            self.chamfer_dist_metric.update(gt, pt_mesh)
            self.precision_recall.update(gt, pt_mesh)

        torch.cuda.empty_cache()  # KEEP THIS FOR MINKOWSKI

    def test_step(self, x: dict, idx):

        outputs = self.forward(x)
        meshes = self.get_meshes(outputs)

        if "VIZ_INT" in self.cfg:
            all_meshes = []
            for aux_i, aux_out in enumerate(outputs["aux_outputs"]):
                meshes = self.get_meshes(outputs)
                all_meshes.append(meshes)

        for batch_idx in range(len(outputs["offsets"])):
            if "extra" not in x.keys():
                gt = x["points"][batch_idx]
            else:
                gt = x["extra"]["gt_points"][batch_idx]

            pt_mesh = meshes[batch_idx]
            pt_mesh = pt_mesh.filter_smooth_taubin(10)
            pt_mesh.compute_vertex_normals()
            in_pcd = o3d.geometry.PointCloud()
            in_pcd.points = o3d.utility.Vector3dVector(x["points"][batch_idx])
            in_pcd.colors = o3d.utility.Vector3dVector(x["colors"][batch_idx])
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(
                x["extra"]["gt_points"][batch_idx])
            gt_pcd.normals = o3d.utility.Vector3dVector(
                x["extra"]["gt_normals"][batch_idx])

            if "VIZ_INT" in self.cfg:
                for int_i, int_mesh in enumerate(all_meshes):
                    gt_pcd = o3d.geometry.PointCloud()
                    gt_pcd.points = o3d.utility.Vector3dVector(gt)
                    final_prediction_lineset = (
                        o3d.geometry.LineSet.create_from_triangle_mesh(
                            int_mesh[batch_idx]
                        )
                    )
                    in_pcd = o3d.geometry.PointCloud()
                    in_pcd.points = o3d.utility.Vector3dVector(
                        x["points"][batch_idx])
                    print("Mesh level", int_i)
                    o3d.visualization.draw_geometries(
                        [final_prediction_lineset, in_pcd],
                        mesh_show_back_face=True,
                        mesh_show_wireframe=True,
                    )

            if "VIZ" in self.cfg:
                gt_pcd = o3d.geometry.PointCloud()
                gt_pcd.points = o3d.utility.Vector3dVector(gt)
                final_prediction_lineset = (
                    o3d.geometry.LineSet.create_from_triangle_mesh(pt_mesh)
                )

                print("final prediction")
                o3d.visualization.draw_geometries(
                    [final_prediction_lineset, in_pcd],
                    mesh_show_back_face=True,
                    mesh_show_wireframe=True,
                )
                o3d.visualization.draw_geometries(
                    [pt_mesh, in_pcd],
                    mesh_show_back_face=True,
                    mesh_show_wireframe=True,
                )

            self.chamfer_dist_metric.update(gt, pt_mesh)
            self.precision_recall.update(gt, pt_mesh)

        torch.cuda.empty_cache()  # KEEP THIS FOR MINKOWSKI

    def validation_epoch_end(self, outputs):
        cd = self.chamfer_dist_metric.compute()
        p, r, f = self.precision_recall.compute_auc()

        self.log("val_chamfer_distance", cd,
                 batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log("val_precision_auc", p, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log("val_recall_auc", r, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log("val_fscore_auc", f, batch_size=self.cfg.TRAIN.BATCH_SIZE)

        self.reset_metrics()
        torch.cuda.empty_cache()  # KEEP THIS FOR MINKOWSKI

    def test_epoch_end(self, outputs):
        cd = self.chamfer_dist_metric.compute()
        p, r, f = self.precision_recall.compute_auc()

        print("------------------------------------")
        print("chamfer distance: {}".format(cd))
        print("computing area under curve")
        print("precision: {}".format(p))
        print("recall: {}".format(r))
        print("fscore: {}".format(f))
        print("------------------------------------")
        p, r, f, _ = self.precision_recall.compute_at_threshold(0.005)
        print("computing at threshold 0.005")
        print("precision: {}".format(p))
        print("recall: {}".format(r))
        print("fscore: {}".format(f))
        print("------------------------------------")
        self.reset_metrics()
        torch.cuda.empty_cache()  # KEEP THIS FOR MINKOWSKI

    def reset_metrics(self):
        self.chamfer_dist_metric.reset()
        self.precision_recall.reset()

    def get_loss(self, inputs: dict, outputs: dict, losses: dict, loss_name: str):

        deformed_template = self.deform_template(outputs=outputs)
        loss_reg_laplace_w, loss_reg_norm_w, loss_cd_w = self.get_weights()

        # regularizers
        if loss_reg_laplace_w > 0 and loss_reg_norm_w > 0:
            deformed_template_mesh = Meshes(
                verts=deformed_template, faces=self.template_faces
            )
            loss_normal = mesh_normal_consistency(deformed_template_mesh)
            loss_laplacian = mesh_laplacian_smoothing(
                deformed_template_mesh, method="uniform"
            )
            losses[loss_name.replace("cd", "reg_normals")
                   ] = loss_normal * loss_reg_norm_w
            losses[loss_name.replace("cd", "reg_laplacian")
                   ] = loss_laplacian * loss_reg_laplace_w
        # ----

        # chamfer
        loss_cd = 0
        for def_tmp, gt_pts in zip(deformed_template, inputs["extra"]["gt_points"]):
            gt = torch.from_numpy(gt_pts).float().cuda().unsqueeze(dim=0)
            loss_pts, _ = self.chamfer_dist(def_tmp.unsqueeze(dim=0), gt)
            loss_cd += loss_pts

        losses[loss_name] = loss_cd * loss_cd_w
        # ----
        return losses

    def deform_template(self, outputs: dict):
        previous_template = outputs["previous_template_points"]
        offsets_scaled = torch.sigmoid(
            outputs["offsets"]) * self.offset_scaling
        deformed_template = previous_template * offsets_scaled

        return deformed_template

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.TRAIN.STEP, gamma=self.cfg.TRAIN.DECAY
        )
        return [optimizer], [scheduler]

    def get_meshes(self, outputs):
        deformed_template = self.deform_template(outputs=outputs)
        meshes = []
        for batch_idx in range(len(outputs["offsets"])):
            pt = deformed_template[batch_idx]
            pt_mesh = o3d.geometry.TriangleMesh()
            pt_mesh.vertices = o3d.utility.Vector3dVector(pt.cpu())
            pt_mesh.triangles = o3d.utility.Vector3iVector(
                self.template_faces[0].cpu())

            template_pcd = o3d.geometry.PointCloud()
            template_pcd.points = o3d.utility.Vector3dVector(
                deformed_template[batch_idx].detach().cpu().numpy())
            colors = np.zeros_like(
                deformed_template[batch_idx].detach().cpu().numpy())

            template_pcd.colors = o3d.utility.Vector3dVector(colors)
            meshes.append(pt_mesh)
        return meshes

    def get_weights(self):
        if 'WEIGHTS_REG_LAP' in self.cfg.LOSS:
            reg_laplace_w = self.cfg.LOSS.WEIGHTS_REG_LAP
        else:
            reg_laplace_w = 1

        if 'WEIGHTS_REG_NOR' in self.cfg.LOSS:
            reg_norm_w = self.cfg.LOSS.WEIGHTS_REG_NOR
        else:
            reg_norm_w = 1

        if 'WEIGHTS_CD' in self.cfg.LOSS:
            cd_w = self.cfg.LOSS.WEIGHTS_CD
        else:
            cd_w = 1

        return reg_laplace_w, reg_norm_w, cd_w
