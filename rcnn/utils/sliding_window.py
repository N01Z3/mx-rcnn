
import numpy as np


def slice_cut(s_predict, s_save, window_size):
    """
    Cut first slice according to second and window size

    Parameters
    ----------
    s_predict: slice
        Zero or positive shifted slice (shift, shift + window_size)
    s_save: slice
        Same as s_predict, but cutted (shift + a, shift + window_size - b)
    window_size: int
        Window size

    Returns
    -------
    slice:
        Slice (a, window_size - b)
    """
    sl = slice(s_save.start - s_predict.start,
               window_size - s_predict.stop + s_save.stop)

    return sl


def sliding_window(im, window_size, stride, cut):
    """
    Find all sliding window positions

    Parameters
    ----------
    im: np.ndarray
        Observed image
    window_size: int
        Window size
    stride: int
        Stride
    cut: int
        Ignore "cut" cells on the left and right

    Returns
    -------
    tuple:
        Slice to save results, slice to cut area for predictions, cnt pixel for mean
    """

    def _steps(im_size):
        """
        Find sliding window positions
        Parameters
        ----------
        im_size: int
            Image size

        Returns
        -------
        list:
            list of steps (0, n-1). -1 will be appended to cover zeros in the end
        """
        res = list(range((im_size - window_size) // stride + 1))
        if (im_size - window_size) % stride != 0:
            res = res + [-1]

        return res

    def _slice(im_size, n, steps, not_full=True):
        """
        Compute slice according to sliding window position

        Parameters
        ----------
        im_size: int
            Image size
        n: int
            Position, 0 - near left border, steps[-1] - near right border,
            steps[-1] == -1 if blank space remained
        steps: list
            All sliding window positions
        not_full:

        Returns
        -------
        slice:
            slice for sliding window position
        """
        if n == -1:
            if not_full:
                sl = slice(steps[-2] * stride + window_size - cut * not_full, im_size)
            else:
                sl = slice(im_size - window_size, im_size)
        elif n == 0:
            sl = slice(0, window_size - cut * not_full)
        elif n == steps[-1]:
            sl = slice(n * stride + cut * not_full, n * stride + window_size)
        else:
            sl = slice(n * stride + cut * not_full, n * stride + window_size - cut * not_full)

        return sl

    positions_save = []
    positions_predict = []

    # mask = np.zeros((im.shape[0], im.shape[1], nlabels))
    cnt = np.zeros(im.shape[:2])

    x_steps, y_steps = _steps(im.shape[1]), _steps(im.shape[0])
    for nx in x_steps:
        for ny in y_steps:
            sx_save, sx_predict = _slice(im.shape[1], nx, x_steps), _slice(im.shape[1], nx, x_steps, not_full=False)
            sy_save, sy_predict = _slice(im.shape[0], ny, y_steps), _slice(im.shape[0], ny, y_steps, not_full=False)

            positions_save.append((sy_save, sx_save))
            positions_predict.append((sy_predict, sx_predict))

            # how-to fill mask with predictions
            # mask[sy_save, sx_save] += predict(im[sy_predict, sx_predict])[slice_cut(sy_predict, sy_save, window_size),
            #                                                               slice_cut(sx_predict, sx_save, window_size)]
            cnt[sy_save, sx_save] += np.ones((window_size, window_size))[slice_cut(sy_predict, sy_save, window_size),
                                                                         slice_cut(sx_predict, sx_save, window_size)]
            #print(sx_save, sx_predict, slice_cut(sx_predict, sx_save, window_size))
    assert cnt.min() != 0, "Wrong params. There are holes in output mask."

    return positions_save, positions_predict, cnt