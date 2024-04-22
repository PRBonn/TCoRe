# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

# # Modified from https://github.com/PRBonn/MaskPLS


import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tcore.utils.interpolate import knn_up


class MinkEncoderDecoder(nn.Module):
    """
    Basic ResNet architecture using sparse convolutions
    """

    def __init__(self, cfg, template_points):
        super().__init__()

        n_classes = 20

        cr = cfg.CR
        self.D = cfg.DIMENSION
        input_dim = cfg.INPUT_DIM
        self.res = cfg.RESOLUTION
        self.interpolate = False  #cfg.INTERPOLATE
        self.feat_key = cfg.FEAT_KEY

        self.template_points = template_points

        if self.interpolate:
            # from tcore.utils.interpolate import knn_up
            self.knn_up = knn_up(cfg.KNN_UP)

        cs = cfg.CHANNELS
        cs = [int(cr*x) for x in cs]
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(input_dim,
                                    cs[0],
                                    kernel_size=3,
                                    stride=1,
                                    dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0],
                                    cs[0],
                                    kernel_size=3,
                                    stride=1,
                                    dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0],
                                  cs[0],
                                  ks=2,
                                  stride=2,
                                  dilation=1,
                                  D=self.D),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1],
                                  cs[1],
                                  ks=2,
                                  stride=2,
                                  dilation=1,
                                  D=self.D),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2],
                                  cs[2],
                                  ks=2,
                                  stride=2,
                                  dilation=1,
                                  D=self.D),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3],
                                  cs[3],
                                  ks=2,
                                  stride=2,
                                  dilation=1,
                                  D=self.D),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3],
                              cs[5],
                              ks=3,
                              stride=1,
                              dilation=1,
                              D=self.D),
                ResidualBlock(cs[5],
                              cs[5],
                              ks=3,
                              stride=1,
                              dilation=1,
                              D=self.D),
            ),
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2],
                              cs[6],
                              ks=3,
                              stride=1,
                              dilation=1,
                              D=self.D),
                ResidualBlock(cs[6],
                              cs[6],
                              ks=3,
                              stride=1,
                              dilation=1,
                              D=self.D),
            ),
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1],
                              cs[7],
                              ks=3,
                              stride=1,
                              dilation=1,
                              D=self.D),
                ResidualBlock(cs[7],
                              cs[7],
                              ks=3,
                              stride=1,
                              dilation=1,
                              D=self.D),
            ),
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0],
                              cs[8],
                              ks=3,
                              stride=1,
                              dilation=1,
                              D=self.D),
                ResidualBlock(cs[8],
                              cs[8],
                              ks=3,
                              stride=1,
                              dilation=1,
                              D=self.D),
            ),
        ])

        self.sem_head = nn.Linear(cs[-1], n_classes)

        levels = [cs[-i] for i in range(4, 0, -1)]
        self.out_bnorm = nn.ModuleList([nn.BatchNorm1d(l) for l in levels])

    def forward(self, x):
        batch_size = len(x['points'])
        in_feats = {}
        in_feats['points'] = x['points']
        if 'normals' in x.keys():
            in_feats['feats'] = [
                np.hstack((x['points'][i], x['normals'][i]))
                for i in range(batch_size)
            ]
        elif 'colors' in x.keys():
            in_feats['feats'] = [
                np.hstack((x['points'][i], x['colors'][i]))
                for i in range(batch_size)
            ]
        else:
            assert False

        in_field = self.TensorField(in_feats)

        x0 = self.stem(in_field.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)

        out_feats = [y1, y2, y3, y4]

        # vox2feat and apply batchnorm
        coors = [
            in_field.decomposed_coordinates for _ in range(len(out_feats))
        ]
        coors = [[c * self.res for c in coors[i]] for i in range(len(coors))]
        if self.interpolate:  # vox2feat using knn interpolation
            bs = in_field.coordinate_manager.number_of_unique_batch_indices()
            vox_coors = [[l.coordinates_at(i) * self.res for i in range(bs)]
                         for l in out_feats]
            feats = [[
                bn(self.knn_up(vox_c, vox_f, pt_c))
                for vox_c, vox_f, pt_c in zip(vc, vf.decomposed_features, pc)
            ] for vc, vf, pc, bn in zip(vox_coors, out_feats, coors,
                                        self.out_bnorm)]
        else:
            out_fields = [feat.slice(in_field) for feat in out_feats]
            feats = [[self.out_bnorm[i](df) for df in f.decomposed_features]
                     for i, f in enumerate(out_fields)]

        feats, coors, pad_masks = self.pad_batch(coors, feats)
        return feats, coors, pad_masks

    def TensorField(self, x):
        """
        Build a tensor field from coordinates and features in the
        input batch
        The coordinates are quantized using the provided resolution

        """

        feat_tfield = ME.TensorField(
            features=torch.from_numpy(np.concatenate(x["feats"], 0)).float(),
            coordinates=ME.utils.batched_coordinates(
                [i / self.res for i in x["points"]], dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.
            UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device="cuda",
        )

        return feat_tfield

    def pad_batch(self, coors, feats):
        """
        From a list of multi-level features create a list of batched tensors with
        point features padded to the max number of points/voxels in the batch.
        For point-level features, create intermediate TensorField:
            voxel_features -> TensorField -> full scale point features

        returns:
            feats: List of batched feature Tensors per feature level (features for each voxel/point)
            coors: List of batched coordinate Tensors per feature level to later generate pos enc
            pad_masks: List of batched bool Tensors indicating where the padding was performed
        """
        # list [feat_levels+1,bs], where each element is [n_pts,dim]
        # 3 features levels + full scale = 4
        # get max number of points/voxels in the batch for each feature level
        maxs = [max([level.shape[0] for level in batch]) for batch in feats]
        # pad to max number of points in batch and put each feature level in a single Tensor
        coors = [
            torch.stack(
                [F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch])
            for i, batch in enumerate(coors)
        ]
        pad_masks = [
            torch.stack([
                F.pad(torch.zeros_like(f[:, 0]), (0, maxs[i] - f.shape[0]),
                      value=1).bool() for f in batch
            ]) for i, batch in enumerate(feats)
        ]
        feats = [
            torch.stack(
                [F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch])
            for i, batch in enumerate(feats)
        ]
        return feats, coors, pad_masks


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    stride=stride,
                                    dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(inc,
                                             outc,
                                             kernel_size=ks,
                                             stride=stride,
                                             dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    stride=stride,
                                    dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(outc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    stride=1,
                                    dimension=D),
            ME.MinkowskiBatchNorm(outc),
        )

        self.downsample = (nn.Sequential() if
                           (inc == outc and stride == 1) else nn.Sequential(
                               ME.MinkowskiConvolution(inc,
                                                       outc,
                                                       kernel_size=1,
                                                       dilation=1,
                                                       stride=stride,
                                                       dimension=D),
                               ME.MinkowskiBatchNorm(outc),
                           ))

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out
