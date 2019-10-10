import torch


def localmax2D(data, min_thresh, min_dist):
    """
    Finds the local maxima for the given data tensor. All computationally intensive
    operations are handled by pytorch on the same device as the given data tensor.

    Parameters:
        data (tensor): the tensor that we search for local maxima. This tensor must
        have at least two dimensions (d1, d2, d3, ..., dn, rows, cols). Each 2D
        (rows, cols) slice will be searched for local maxima. If neighboring pixels
        have the same value we follow a simple tie breaking algorithm. The tie will
        be broken by considering the pixel with the larger row greater, or if the
        rows are also equal, the pixel with the larger column index is considered
        greater.

        min_thresh (number): any pixels below this threshold will be excluded
        from the local maxima search

        min_dist (integer): minimum neighborhood size in pixels. We search a square
        neighborhood around each pixel min_dist pixels out. If a pixel is the largest
        value in this neighborhood we mark it a local maxima. All pixels within
        min_dist of the 2D boundary are excluded from the local maxima search. This
        parameter must be >= 1.

    Returns:
        A boolean tensor with the same shape as data and on the same device. Elements
        will be True where a local maxima was detected and False elsewhere.
    """

    assert min_dist >= 1

    data_size = list(data.size())

    accum_mask = data >= min_thresh

    # mask out a frame around the 2D space the size of min_dist
    accum_mask[..., :min_dist, :] = False
    accum_mask[..., -min_dist:, :] = False
    accum_mask[..., :min_dist] = False
    accum_mask[..., -min_dist:] = False

    for row_offset in range(-min_dist, min_dist + 1):
        for col_offset in range(-min_dist, min_dist + 1):

            # nothing to do if we're at 0, 0
            if row_offset != 0 or col_offset != 0:

                offset_data = data
                if row_offset < 0:
                    offset_data = offset_data[..., :row_offset, :]
                    padding_size = data_size.copy()
                    padding_size[-2] = -row_offset
                    padding = offset_data.new_empty(padding_size)

                    offset_data = torch.cat([padding, offset_data], -2)
                elif row_offset > 0:
                    offset_data = offset_data[..., row_offset:, :]
                    padding_size = data_size.copy()
                    padding_size[-2] = row_offset
                    padding = offset_data.new_empty(padding_size)

                    offset_data = torch.cat([offset_data, padding], -2)

                if col_offset < 0:
                    offset_data = offset_data[..., :col_offset]
                    padding_size = data_size.copy()
                    padding_size[-1] = -col_offset
                    padding = offset_data.new_empty(padding_size)

                    offset_data = torch.cat([padding, offset_data], -1)
                elif col_offset > 0:
                    offset_data = offset_data[..., col_offset:]
                    padding_size = data_size.copy()
                    padding_size[-1] = col_offset
                    padding = offset_data.new_empty(padding_size)

                    offset_data = torch.cat([offset_data, padding], -1)

                # dominance will act as a "tie breaker" for pixels that have equal value
                data_is_dominant = False
                if row_offset != 0:
                    data_is_dominant = row_offset > 0
                else:
                    data_is_dominant = col_offset > 0

                if data_is_dominant:
                    accum_mask &= data >= offset_data
                else:
                    accum_mask &= data > offset_data

    return accum_mask


def flatten_indices(indices_2d, shape):
    """
    This function will "flatten" the given index matrix such that it can be used
    as input to pytorch's `take` function.
    """

    # calculate the index multiplier vector that allows us to convert
    # from vector indicies to flat indices
    index_mult_vec = indices_2d.new_ones(len(shape))
    for i in range(len(shape) - 1):
        index_mult_vec[ : i + 1] *= shape[i + 1]

    return torch.sum(indices_2d * index_mult_vec, dim=1)
