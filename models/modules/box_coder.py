import torch
import torch.nn as nn
from typing import Optional

from timm.layers import to_3tuple

__all__ = ["PointWHD", "PatchCenterOffset"]


class PointWHD(nn.Module):
    """Point With Height, Width, and Depth (WHD) Anchoring Layer.

    This class generates anchors in a 3D space based on input size and patch
    counts, and allows for decoding deformation logits to obtain corresponding
    deformation centers. The anchors represent the center points of the patches
    in the input space, which can be used in tasks like deformable convolution
    or region proposal in 3D data.

    Parameters
    ----------
    input_size : tuple[int, int, int]
        Size of the input tensor in the form (depth, height, width).
    patch_count : tuple[int, int, int]
        Number of patches to divide the input into in each dimension.
    weights : tuple[float, float, float], optional
        Scaling factors for deformation logits in each dimension. Default is None.
    tanh : bool, optional
        If True, applies the hyperbolic tangent function to the logits for scaling.
        Default is None.
    """

    def __init__(
        self,
        input_size: tuple[int, int, int],
        patch_count: tuple[int, int, int],
        weights: Optional[tuple[float, float, float]] = None,
        tanh: Optional[bool] = None,
    ):
        super().__init__()
        self.input_size = to_3tuple(input_size)
        self.patch_count = to_3tuple(patch_count)
        self.weights = weights
        self._generate_anchor()  # Generate anchors based on patch count
        self.tanh = tanh

    def _generate_anchor(self) -> None:
        """Generate anchors for the patch centers."""
        anchors = []
        patch_stride = [1.0 / p for p in self.patch_count]
        for i in range(self.patch_count[0]):
            for j in range(self.patch_count[1]):
                for k in range(self.patch_count[2]):
                    x = (0.5 + i) * patch_stride[0]
                    y = (0.5 + j) * patch_stride[1]
                    z = (0.5 + k) * patch_stride[2]
                    anchors.append([x, y, z])
        anchors = torch.as_tensor(anchors)  # Convert to tensor of shape [N, 3]
        self.register_buffer("anchor", anchors)  # Register anchors as a buffer

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self, deform_logits: torch.Tensor, model_offset: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to decode deformation logits into centers.

        Parameters
        ----------
        deform_logits : torch.Tensor
            Input deformation logits with shape [B, N, 3].
        model_offset : torch.Tensor = None
            An optional model offset (expected to be None).

        Returns
        -------
        torch.Tensor
            Deformed centers in the 3D space.
        """
        assert model_offset is None  # Ensure model_offset is None
        deform_centers = self.decode(deform_logits)  # Decode logits to centers
        return deform_centers

    def decode(self, deform_logits: torch.Tensor) -> torch.Tensor:
        """
        Decode deformation logits into deformation centers.

        Parameters
        ----------
        deform_logits : torch.Tensor
            Input deformation logits with shape [B, N, 3].

        Returns
        -------
        torch.tensor:
            Deformed centers with shape [B, N, 3].
        """
        anchor = self.anchor
        pixel = [1.0 / p for p in self.patch_count]
        wx, wy, wz = self.weights

        # Apply scaling based on tanh or linear
        dx = (
            torch.tanh(deform_logits[:, :, 0] / wx) * pixel[0]
            if self.tanh
            else deform_logits[:, :, 0] * pixel[0] / wx
        )
        dy = (
            torch.tanh(deform_logits[:, :, 1] / wy) * pixel[1]
            if self.tanh
            else deform_logits[:, :, 1] * pixel[1] / wy
        )
        dz = (
            torch.tanh(deform_logits[:, :, 2] / wz) * pixel[2]
            if self.tanh
            else deform_logits[:, :, 2] * pixel[2] / wz
        )

        deform_centers = torch.zeros_like(deform_logits)
        ref_x = anchor[:, 0].unsqueeze(0)
        ref_y = anchor[:, 1].unsqueeze(0)
        ref_z = anchor[:, 2].unsqueeze(0)

        # Calculate final deform centers
        deform_centers[:, :, 0] = dx + ref_x
        deform_centers[:, :, 1] = dy + ref_y
        deform_centers[:, :, 2] = dz + ref_z
        deform_centers = torch.clamp(deform_centers, min=0.0, max=1.0)

        return deform_centers

    def get_offsets(self, deform_centers: torch.Tensor) -> torch.Tensor:
        """
        Calculate offsets based on the deformation centers.

        Parameters
        ----------
        deform_centers : torch.Tensor
            Deformation centers with shape [B, N, 3].

        Returns
        -------
        torch.Tensor
            Offsets relative to the anchors.
        """
        return (deform_centers - self.anchor) * torch.as_tensor(
            self.input_size, device=deform_centers.device
        ).unsqueeze(0).unsqueeze(0)


class PatchCenterOffset(PointWHD):
    """Patch Center Offset Layer.

    This class extends the PointWHD class to calculate the offsets for patch centers
    based on deformation logits, allowing for more flexible manipulation of
    spatial data in a 3D context. It generates bounding boxes for patches and
    supports operations such as scaling and meshgrid creation.

    Parameters
    ----------
    input_size : tuple[int, int, int]
        Size of the input tensor in the form (depth, height, width).
    patch_count : tuple[int, int, int]
        Number of patches to divide the input into in each dimension.
    weights : tuple[float, float, float], optional
        Scaling factors for deformation logits in each dimension. Default is (2.0, 2.0, 2.0).
    pts : tuple[int, int, int], optional
        Number of points in each dimension for meshgrid. Default is (1, 1, 1).
    tanh : bool, optional
        If True, applies the hyperbolic tangent function to the logits. Default is True.
    """

    def __init__(
        self,
        input_size: tuple[int, int, int],
        patch_count: tuple[int, int, int],
        weights: Optional[tuple[float, float, float]] = (2.0, 2.0, 2.0),
        pts: Optional[tuple[int, int, int]] = (1, 1, 1),
        tanh: Optional[bool] = True,
    ):
        super().__init__(
            input_size=input_size, patch_count=patch_count, weights=weights, tanh=tanh
        )
        self.patch_pixel = pts
        self.scale_bias = None
        self.count = 0

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self, deform_logits: torch.Tensor, model_offset: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to decode deformation logits into points.

        Parameters
        ----------
        deform_logits : torch.Tensor
            Input deformation logits with shape [B, N, 3].
        model_offset : torch.Tensor = None
            An optional model offset (expected to be None).

        Returns
        -------
        torch.Tensor
            Points generated based on deformation boxes.
        """
        assert model_offset is None  # Ensure model_offset is None
        deform_boxes = self.decode(deform_logits)  # Decode logits to boxes
        points = self.meshgrid(deform_boxes)  # Generate points in 3D space
        return points

    def decode(self, deform_logits: torch.Tensor) -> torch.Tensor:
        """
        Decode deformation logits into bounding boxes.

        Parameters
        ----------
        deform_logits : torch.Tensor
            Input deformation logits with shape [B, N, 3].

        Returns
        -------
        torch.Tensor
            Bounding boxes with shape [B, N, 6].
        """
        anchor = self.anchor
        pixel = [1.0 / p for p in self.patch_count]
        wx, wy, wz = self.weights

        # Apply scaling based on tanh or linear
        dx = (
            torch.tanh(deform_logits[:, :, 0] / wx) * pixel[0] / 2
            if self.tanh
            else deform_logits[:, :, 0] * pixel[0] / wx / 2
        )
        dy = (
            torch.tanh(deform_logits[:, :, 1] / wy) * pixel[1] / 2
            if self.tanh
            else deform_logits[:, :, 1] * pixel[1] / wy / 2
        )
        dz = (
            torch.tanh(deform_logits[:, :, 2] / wz) * pixel[2] / 2
            if self.tanh
            else deform_logits[:, :, 2] * pixel[2] / wz / 2
        )

        b, n, _ = deform_logits.shape
        deform_boxes = torch.zeros((b, n, 6), device=deform_logits.device)

        # Calculate the center points and bounding boxes
        ref_x = anchor[:, 0].unsqueeze(0)
        ref_y = anchor[:, 1].unsqueeze(0)
        ref_z = anchor[:, 2].unsqueeze(0)

        center_x = torch.clamp(dx + ref_x, min=pixel[0] / 2, max=1.0 - pixel[0] / 2)
        center_y = torch.clamp(dy + ref_y, min=pixel[1] / 2, max=1.0 - pixel[1] / 2)
        center_z = torch.clamp(dz + ref_z, min=pixel[2] / 2, max=1.0 - pixel[2] / 2)

        self.deform_centers = [center_x.cpu(), center_y.cpu(), center_z.cpu()]

        deform_boxes[:, :, 0] = center_x - pixel[0] / 2
        deform_boxes[:, :, 1] = center_y - pixel[1] / 2
        deform_boxes[:, :, 2] = center_z - pixel[2] / 2
        deform_boxes[:, :, 3] = center_x + pixel[0] / 2
        deform_boxes[:, :, 4] = center_y + pixel[1] / 2
        deform_boxes[:, :, 5] = center_z + pixel[2] / 2

        return deform_boxes

    def get_offsets(self, deform_boxes: torch.Tensor) -> torch.Tensor:
        """
        Calculate offsets based on the deformation boxes.

        Parameters
        ----------
        deform_boxes : torch.Tensor
            Deformation boxes with shape [B, N, 6].

        Returns
        -------
        torch.Tensor
            Offsets relative to the anchors.
        """
        return (
            deform_boxes - self.anchor.repeat(1, 2)  # shape: (B, N, 6)
        ) * torch.as_tensor(self.input_size, device=deform_boxes.device).unsqueeze(
            0
        ).unsqueeze(0)

    def get_scales(self, deform_boxes: torch.Tensor) -> torch.Tensor:
        """
        Calculate scales based on the deformation boxes.

        Parameters
        ----------
        deform_boxes : torch.Tensor
            Deformation boxes with shape [B, N, 6].

        Returns
        -------
        torch.Tensor
            Scales of the boxes based on the input size.
        """
        return (deform_boxes[:, :, 3:] - deform_boxes[:, :, :3]) * torch.as_tensor(
            self.input_size, device=deform_boxes.device
        ).unsqueeze(0).unsqueeze(0)

    def meshgrid(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Generate a meshgrid based on the given bounding boxes.

        Parameters
        ----------
        boxes : torch.Tensor
            Bounding boxes with shape [B, N, 6].

        Returns
        -------
        torch.Tensor
            Meshgrid of points based on the boxes with shape [B, N, D, H, W].
        """
        xs, ys, zs = boxes[:, :, 0::3], boxes[:, :, 1::3], boxes[:, :, 2::3]

        xs = torch.nn.functional.interpolate(
            xs, size=self.patch_pixel[0], mode="linear", align_corners=True
        )
        ys = torch.nn.functional.interpolate(
            ys, size=self.patch_pixel[1], mode="linear", align_corners=True
        )
        zs = torch.nn.functional.interpolate(
            zs, size=self.patch_pixel[2], mode="linear", align_corners=True
        )

        # Create the meshgrid
        xs = (
            xs.unsqueeze(3)
            .repeat_interleave(self.patch_pixel[1], dim=3)
            .unsqueeze(4)
            .repeat_interleave(self.patch_pixel[2], dim=4)
        )
        ys = (
            ys.unsqueeze(2)
            .repeat_interleave(self.patch_pixel[0], dim=2)
            .unsqueeze(4)
            .repeat_interleave(self.patch_pixel[2], dim=4)
        )
        zs = (
            zs.unsqueeze(2)
            .repeat_interleave(self.patch_pixel[0], dim=2)
            .unsqueeze(3)
            .repeat_interleave(self.patch_pixel[1], dim=3)
        )

        results = torch.stack([xs, ys, zs], dim=-1)  # Stack the coordinates
        return results
