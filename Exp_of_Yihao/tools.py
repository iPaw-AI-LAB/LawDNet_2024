import os
import numpy as np
import cv2
import onnx
import onnxruntime
import time
import textwrap
import torch
import math
from math import ceil
from itertools import product as product
class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source feature map.
    """
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
    
# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def batch_decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:,:, :2] + loc[:,:, :2] * variances[0] * priors[:,:, 2:],
        priors[:,:, 2:] * torch.exp(loc[:,:, 2:] * variances[1])), -1)
    boxes[:,:, :2] -= boxes[:,:, 2:] / 2
    boxes[:,:, 2:] += boxes[:,:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def _umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array_like
        Source coordinates.
    dst : (M, N) array_like
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`

    """
    src = np.asarray(src)
    dst = np.asarray(dst)

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T

def _euler_rotation(axis, angle):
    """Produce a single-axis Euler rotation matrix.

    Parameters
    ----------
    axis : int in {0, 1, 2}
        The axis of rotation.
    angle : float
        The angle of rotation in radians.

    Returns
    -------
    Ri : array of float, shape (3, 3)
        The rotation matrix along axis `axis`.
    """
    i = axis
    s = (-1)**i * np.sin(angle)
    c = np.cos(angle)
    R2 = np.array([[c, -s],
                   [s,  c]])
    Ri = np.eye(3)
    # We need the axes other than the rotation axis, in the right order:
    # 0 -> (1, 2); 1 -> (0, 2); 2 -> (0, 1).
    axes = sorted({0, 1, 2} - {axis})
    # We then embed the 2-axis rotation matrix into the full matrix.
    # (1, 2) -> R[1:3:1, 1:3:1] = R2, (0, 2) -> R[0:3:2, 0:3:2] = R2, etc.
    sl = slice(axes[0], axes[1] + 1, axes[1] - axes[0])
    Ri[sl, sl] = R2
    return Ri

def _euler_rotation_matrix(angles, axes=None):
    """Produce an Euler rotation matrix from the given angles.

    The matrix will have dimension equal to the number of angles given.

    Parameters
    ----------
    angles : array of float, shape (3,)
        The transformation angles in radians.
    axes : list of int
        The axes about which to produce the rotation. Defaults to 0, 1, 2.

    Returns
    -------
    R : array of float, shape (3, 3)
        The Euler rotation matrix.
    """
    if axes is None:
        axes = range(3)
    dim = len(angles)
    R = np.eye(dim)
    for i, angle in zip(axes, angles):
        R = R @ _euler_rotation(i, angle)
    return R


def _center_and_normalize_points(points):
    """Center and normalize image points.

    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.

    Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(D).

    If the points are all identical, the returned values will contain nan.

    Parameters
    ----------
    points : (N, D) array
        The coordinates of the image points.

    Returns
    -------
    matrix : (D+1, D+1) array_like
        The transformation matrix to obtain the new points.
    new_points : (N, D) array
        The transformed image points.

    References
    ----------
    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."
           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6
           (1997): 580-593.

    """
    n, d = points.shape
    centroid = np.mean(points, axis=0)

    centered = points - centroid
    rms = np.sqrt(np.sum(centered ** 2) / n)

    # if all the points are the same, the transformation matrix cannot be
    # created. We return an equivalent matrix with np.nans as sentinel values.
    # This obviates the need for try/except blocks in functions calling this
    # one, and those are only needed when actual 0 is reached, rather than some
    # small value; ie, we don't need to worry about numerical stability here,
    # only actual 0.
    if rms == 0:
        return np.full((d + 1, d + 1), np.nan), np.full_like(points, np.nan)

    norm_factor = np.sqrt(d) / rms

    part_matrix = norm_factor * np.concatenate(
            (np.eye(d), -centroid[:, np.newaxis]), axis=1
            )
    matrix = np.concatenate(
            (part_matrix, [[0,] * d + [1]]), axis=0
            )

    points_h = np.row_stack([points.T, np.ones(n)])

    new_points_h = (matrix @ points_h).T

    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]

    return matrix, new_points

def get_bound_method_class(m):
    """Return the class for a bound method.

    """
    return m.im_class if sys.version < '3' else m.__self__.__class__


class GeometricTransform:
    """Base class for geometric transformations.

    """
    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array_like
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Destination coordinates.

        """
        raise NotImplementedError()

    def inverse(self, coords):
        """Apply inverse transformation.

        Parameters
        ----------
        coords : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        coords : (N, 2) array
            Source coordinates.

        """
        raise NotImplementedError()

    def residuals(self, src, dst):
        """Determine residuals of transformed destination coordinates.

        For each transformed source coordinate the euclidean distance to the
        respective destination coordinate is determined.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        residuals : (N, ) array
            Residual for coordinate.

        """
        return np.sqrt(np.sum((self(src) - dst)**2, axis=1))

    def __add__(self, other):
        """Combine this transformation with another.

        """
        raise NotImplementedError()

class ProjectiveTransform(GeometricTransform):
    r"""Projective transformation.

    Apply a projective transformation (homography) on coordinates.

    For each homogeneous coordinate :math:`\mathbf{x} = [x, y, 1]^T`, its
    target position is calculated by multiplying with the given matrix,
    :math:`H`, to give :math:`H \mathbf{x}`::

      [[a0 a1 a2]
       [b0 b1 b2]
       [c0 c1 1 ]].

    E.g., to rotate by theta degrees clockwise, the matrix should be::

      [[cos(theta) -sin(theta) 0]
       [sin(theta)  cos(theta) 0]
       [0            0         1]]

    or, to translate x by 10 and y by 20::

      [[1 0 10]
       [0 1 20]
       [0 0 1 ]].

    Parameters
    ----------
    matrix : (D+1, D+1) array_like, optional
        Homogeneous transformation matrix.
    dimensionality : int, optional
        The number of dimensions of the transform. This is ignored if
        ``matrix`` is not None.

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    """

    def __init__(self, matrix=None, *, dimensionality=2):
        if matrix is None:
            # default to an identity transform
            matrix = np.eye(dimensionality + 1)
        else:
            matrix = np.asarray(matrix)
            dimensionality = matrix.shape[0] - 1
            if matrix.shape != (dimensionality + 1, dimensionality + 1):
                raise ValueError("invalid shape of transformation matrix")
        self.params = matrix
        self._coeffs = range(matrix.size - 1)

    @property
    def _inv_matrix(self):
        return np.linalg.inv(self.params)

    def _apply_mat(self, coords, matrix):
        ndim = matrix.shape[0] - 1
        coords = np.array(coords, copy=False, ndmin=2)

        src = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
        dst = src @ matrix.T

        # below, we will divide by the last dimension of the homogeneous
        # coordinate matrix. In order to avoid division by zero,
        # we replace exact zeros in this column with a very small number.
        dst[dst[:, ndim] == 0, ndim] = np.finfo(float).eps
        # rescale to homogeneous coordinates
        dst[:, :ndim] /= dst[:, ndim:ndim+1]

        return dst[:, :ndim]

    def __array__(self, dtype=None):
        if dtype is None:
            return self.params
        else:
            return self.params.astype(dtype)

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, D) array_like
            Source coordinates.

        Returns
        -------
        coords_out : (N, D) array
            Destination coordinates.

        """
        return self._apply_mat(coords, self.params)

    def inverse(self, coords):
        """Apply inverse transformation.

        Parameters
        ----------
        coords : (N, D) array_like
            Destination coordinates.

        Returns
        -------
        coords_out : (N, D) array
            Source coordinates.

        """
        return self._apply_mat(coords, self._inv_matrix)

    def estimate(self, src, dst, weights=None):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = (a0*x + a1*y + a2) / (c0*x + c1*y + 1)
            Y = (b0*x + b1*y + b2) / (c0*x + c1*y + 1)

        These equations can be transformed to the following form::

            0 = a0*x + a1*y + a2 - c0*x*X - c1*y*X - X
            0 = b0*x + b1*y + b2 - c0*x*Y - c1*y*Y - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[x y 1 0 0 0 -x*X -y*X -X]
                   [0 0 0 x y 1 -x*Y -y*Y -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 a2 b0 b1 b2 c0 c1 c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        Weights can be applied to each pair of corresponding points to
        indicate, particularly in an overdetermined system, if point pairs have
        higher or lower confidence or uncertainties associated with them. From
        the matrix treatment of least squares problems, these weight values are
        normalised, square-rooted, then built into a diagonal matrix, by which
        A is multiplied.

        In case of the affine transformation the coefficients c0 and c1 are 0.
        Thus the system of equations is::

            A   = [[x y 1 0 0 0 -X]
                   [0 0 0 x y 1 -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 a2 b0 b1 b2 c3]

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.
        weights : (N,) array_like, optional
            Relative weight values for each pair of points.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        src = np.asarray(src)
        dst = np.asarray(dst)
        n, d = src.shape

        src_matrix, src = _center_and_normalize_points(src)
        dst_matrix, dst = _center_and_normalize_points(dst)
        if not np.all(np.isfinite(src_matrix + dst_matrix)):
            self.params = np.full((d + 1, d + 1), np.nan)
            return False

        # params: a0, a1, a2, b0, b1, b2, c0, c1
        A = np.zeros((n * d, (d+1) ** 2))
        # fill the A matrix with the appropriate block matrices; see docstring
        # for 2D example — this can be generalised to more blocks in the 3D and
        # higher-dimensional cases.
        for ddim in range(d):
            A[ddim*n : (ddim+1)*n, ddim*(d+1) : ddim*(d+1) + d] = src
            A[ddim*n : (ddim+1)*n, ddim*(d+1) + d] = 1
            A[ddim*n : (ddim+1)*n, -d-1:-1] = src
            A[ddim*n : (ddim+1)*n, -1] = -1
            A[ddim*n : (ddim+1)*n, -d-1:] *= -dst[:, ddim:(ddim+1)]

        # Select relevant columns, depending on params
        A = A[:, list(self._coeffs) + [-1]]

        # Get the vectors that correspond to singular values, also applying
        # the weighting if provided
        if weights is None:
            _, _, V = np.linalg.svd(A)
        else:
            weights = np.asarray(weights)
            W = np.diag(np.tile(np.sqrt(weights / np.max(weights)), d))
            _, _, V = np.linalg.svd(W @ A)

        # if the last element of the vector corresponding to the smallest
        # singular value is close to zero, this implies a degenerate case
        # because it is a rank-defective transform, which would map points
        # to a line rather than a plane.
        if np.isclose(V[-1, -1], 0):
            self.params = np.full((d + 1, d + 1), np.nan)
            return False

        H = np.zeros((d+1, d+1))
        # solution is right singular vector that corresponds to smallest
        # singular value
        H.flat[list(self._coeffs) + [-1]] = - V[-1, :-1] / V[-1, -1]
        H[d, d] = 1

        # De-center and de-normalize
        H = np.linalg.inv(dst_matrix) @ H @ src_matrix

        # Small errors can creep in if points are not exact, causing the last
        # element of H to deviate from unity. Correct for that here.
        H /= H[-1, -1]

        self.params = H

        return True

    def __add__(self, other):
        """Combine this transformation with another."""
        if isinstance(other, ProjectiveTransform):
            # combination of the same types result in a transformation of this
            # type again, otherwise use general projective transformation
            if type(self) == type(other):
                tform = self.__class__
            else:
                tform = ProjectiveTransform
            return tform(other.params @ self.params)
        elif (hasattr(other, '__name__')
                and other.__name__ == 'inverse'
                and hasattr(get_bound_method_class(other), '_inv_matrix')):
            return ProjectiveTransform(other.__self__._inv_matrix @ self.params)
        else:
            raise TypeError("Cannot combine transformations of differing "
                            "types.")

    def __nice__(self):
        """common 'paramstr' used by __str__ and __repr__"""
        npstring = np.array2string(self.params, separator=', ')
        paramstr = 'matrix=\n' + textwrap.indent(npstring, '    ')
        return paramstr

    def __repr__(self):
        """Add standard repr formatting around a __nice__ string"""
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return f'<{classstr}({paramstr}) at {hex(id(self))}>'

    def __str__(self):
        """Add standard str formatting around a __nice__ string"""
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return f'<{classstr}({paramstr})>'

    @property
    def dimensionality(self):
        """The dimensionality of the transformation."""
        return self.params.shape[0] - 1

class EuclideanTransform(ProjectiveTransform):
    """Euclidean transformation, also known as a rigid transform.

    Has the following form::

        X = a0 * x - b0 * y + a1 =
          = x * cos(rotation) - y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = x * sin(rotation) + y * cos(rotation) + b1

    where the homogeneous transformation matrix is::

        [[a0  b0  a1]
         [b0  a0  b1]
         [0   0    1]]

    The Euclidean transformation is a rigid transformation with rotation and
    translation parameters. The similarity transformation extends the Euclidean
    transformation with a single scaling factor.

    Parameters
    ----------
    matrix : (D+1, D+1) array_like, optional
        Homogeneous transformation matrix.
    rotation : float or sequence of float, optional
        Rotation angle in counter-clockwise direction as radians. If given as
        a vector, it is interpreted as Euler rotation angles [1]_. Only 2D
        (single rotation) and 3D (Euler rotations) values are supported. For
        higher dimensions, you must provide or estimate the transformation
        matrix.
    translation : sequence of float, length D, optional
        Translation parameters for each axis.
    dimensionality : int, optional
        The dimensionality of the transform.

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    """

    def __init__(self, matrix=None, rotation=None, translation=None,
                 *, dimensionality=2):
        params_given = rotation is not None or translation is not None

        if params_given and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix
        elif params_given:
            if rotation is None:
                dimensionality = len(translation)
                if dimensionality == 2:
                    rotation = 0
                elif dimensionality == 3:
                    rotation = np.zeros(3)
                else:
                    raise ValueError(
                        'Parameters cannot be specified for dimension '
                        f'{dimensionality} transforms'
                    )
            else:
                if not np.isscalar(rotation) and len(rotation) != 3:
                    raise ValueError(
                        'Parameters cannot be specified for dimension '
                        f'{dimensionality} transforms'
                    )
            if translation is None:
                translation = (0,) * dimensionality

            if dimensionality == 2:
                self.params = np.array([
                    [math.cos(rotation), - math.sin(rotation), 0],
                    [math.sin(rotation),   math.cos(rotation), 0],
                    [                 0,                    0, 1]
                ])
            elif dimensionality == 3:
                self.params = np.eye(dimensionality + 1)
                self.params[:dimensionality, :dimensionality] = (
                    _euler_rotation_matrix(rotation)
                )
            self.params[0:dimensionality, dimensionality] = translation
        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        self.params = _umeyama(src, dst, False)

        # _umeyama will return nan if the problem is not well-conditioned.
        return not np.any(np.isnan(self.params))

    @property
    def rotation(self):
        if self.dimensionality == 2:
            return math.atan2(self.params[1, 0], self.params[1, 1])
        elif self.dimensionality == 3:
            # Returning 3D Euler rotation matrix
            return self.params[:3, :3]
        else:
            raise NotImplementedError(
                'Rotation only implemented for 2D and 3D transforms.'
            )

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]

class SimilarityTransform(EuclideanTransform):
    """Similarity transformation.

    Has the following form in 2D::

        X = a0 * x - b0 * y + a1 =
          = s * x * cos(rotation) - s * y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = s * x * sin(rotation) + s * y * cos(rotation) + b1

    where ``s`` is a scale factor and the homogeneous transformation matrix is::

        [[a0  b0  a1]
         [b0  a0  b1]
         [0   0    1]]

    The similarity transformation extends the Euclidean transformation with a
    single scaling factor in addition to the rotation and translation
    parameters.

    Parameters
    ----------
    matrix : (dim+1, dim+1) array_like, optional
        Homogeneous transformation matrix.
    scale : float, optional
        Scale factor. Implemented only for 2D and 3D.
    rotation : float, optional
        Rotation angle in counter-clockwise direction as radians.
        Implemented only for 2D and 3D. For 3D, this is given in ZYX Euler
        angles.
    translation : (dim,) array_like, optional
        x, y[, z] translation parameters. Implemented only for 2D and 3D.

    Attributes
    ----------
    params : (dim+1, dim+1) array
        Homogeneous transformation matrix.

    """

    def __init__(self, matrix=None, scale=None, rotation=None,
                 translation=None, *, dimensionality=2):
        self.params = None
        params = any(param is not None
                     for param in (scale, rotation, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                self.params = matrix
                dimensionality = matrix.shape[0] - 1
        if params:
            if dimensionality not in (2, 3):
                raise ValueError('Parameters only supported for 2D and 3D.')
            matrix = np.eye(dimensionality + 1, dtype=float)
            if scale is None:
                scale = 1
            if rotation is None:
                rotation = 0 if dimensionality == 2 else (0, 0, 0)
            if translation is None:
                translation = (0,) * dimensionality
            if dimensionality == 2:
                ax = (0, 1)
                c, s = np.cos(rotation), np.sin(rotation)
                matrix[ax, ax] = c
                matrix[ax, ax[::-1]] = -s, s
            else:  # 3D rotation
                matrix[:3, :3] = _euler_rotation_matrix(rotation)

            matrix[:dimensionality, :dimensionality] *= scale
            matrix[:dimensionality, dimensionality] = translation
            self.params = matrix
        elif self.params is None:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        self.params = _umeyama(src, dst, estimate_scale=True)

        # _umeyama will return nan if the problem is not well-conditioned.
        return not np.any(np.isnan(self.params))

    @property
    def scale(self):
        # det = scale**(# of dimensions), therefore scale = det**(1/ndim)
        if self.dimensionality == 2:
            return np.sqrt(np.linalg.det(self.params))
        elif self.dimensionality == 3:
            return np.cbrt(np.linalg.det(self.params))
        else:
            raise NotImplementedError(
                'Scale is only implemented for 2D and 3D.')

def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = SimilarityTransform(rotation=rot)
    t4 = SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M

def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts








def umeyama(src, dst, estimate_scale=True):
    """
    Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    Reference
    Least-squares estimation of transformation parameters between two point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """
    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T[:2]

def transform_points(mat, points):
    if not isinstance(points, np.ndarray):
        points = np.float32(points)

    dtype = points.dtype

    points = np.pad(points, ((0,0), (0,1) ), constant_values=(1,), mode='constant')

    return np.matmul( np.concatenate( [ mat, [[0,0,1]] ], 0), points.T).T[:,:2].astype(dtype)

def invert(mat):
    """
    returns inverted Affine2DMat
    """
    ((a, b, c),
        (d, e, f)) = mat
    D = a*e - b*d
    D = 1.0 / D if D != 0.0 else 0.0
    a, b, c, d, e, f = ( e*D, -b*D, (b*f-e*c)*D ,
                        -d*D,  a*D, (d*c-a*f)*D )
    return np.array([[a, b, c],[d, e, f]])

def segment_to_vector(p1 : np.ndarray, p2 : np.ndarray):
    """
        p1  (2,)
        p2  (2,)
    """
    x = p2-p1
    x /= np.linalg.norm(x)
    return x

def segment_length(p1 : np.ndarray, p2 : np.ndarray):
    """
        p1  (2,)
        p2  (2,)
    """
    return np.linalg.norm(p2-p1)
