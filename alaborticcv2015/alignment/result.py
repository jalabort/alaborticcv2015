from __future__ import division
import abc
import numpy as np
from menpo.transform import Scale
from menpo.image import Image
from menpofit.fittingresult import compute_error


class Result(object):

    @abc.abstractproperty
    def n_iters(self):
        r"""
        Returns the number of iterations.
        """

    @abc.abstractmethod
    def shapes(self, as_points=False):
        r"""
        Generates a list containing the shapes obtained at each fitting
        iteration.

        Parameters
        -----------
        as_points : boolean, optional
            Whether the results is returned as a list of :map:`PointCloud`s or
            ndarrays.

            Default: `False`

        Returns
        -------
        shapes : :map:`PointCloud`s or ndarray list
            A list containing the shapes obtained at each fitting iteration.
        """

    @abc.abstractproperty
    def final_shape(self):
        r"""
        Returns the final fitted shape.
        """

    @abc.abstractproperty
    def initial_shape(self):
        r"""
        Returns the initial shape from which the fitting started.
        """

    @property
    def gt_shape(self):
        r"""
        Returns the original ground truth shape associated to the image.
        """
        return self._gt_shape

    @property
    def fitted_image(self):
        r"""
        Returns a copy of the fitted image with the following landmark
        groups attached to it:
            - ``initial``, containing the initial fitted shape .
            - ``final``, containing the final shape.
            - ``ground``, containing the ground truth shape. Only returned if
            the ground truth shape was provided.

        :type: :map:`Image`
        """
        image = Image(self.image.pixels)

        image.landmarks['initial'] = self.initial_shape
        image.landmarks['final'] = self.final_shape
        if self.gt_shape is not None:
            image.landmarks['ground'] = self.gt_shape
        return image

    @property
    def iter_image(self):
        r"""
        Returns a copy of the fitted image with a as many landmark groups as
        iteration run by fitting procedure:
            - ``iter_0``, containing the initial shape.
            - ``iter_1``, containing the the fitted shape at the first
            iteration.
            - ``...``
            - ``iter_n``, containing the final fitted shape.

        :type: :map:`Image`
        """
        image = Image(self.image.pixels)
        for j, s in enumerate(self.shapes()):
            image.landmarks['iter_'+str(j)] = s
        return image

    def errors(self, error_type='me_norm'):
        r"""
        Returns a list containing the error at each fitting iteration.

        Parameters
        -----------
        error_type : `str` ``{'me_norm', 'me', 'rmse'}``, optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

        Returns
        -------
        errors : `list` of `float`
            The errors at each iteration of the fitting process.
        """
        if self.gt_shape is not None:
            return [compute_error(t, self.gt_shape, error_type)
                    for t in self.shapes()]
        else:
            raise ValueError('Ground truth has not been set, errors cannot '
                             'be computed')

    def final_error(self, error_type='me_norm'):
        r"""
        Returns the final fitting error.

        Parameters
        -----------
        error_type : `str` ``{'me_norm', 'me', 'rmse'}``, optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

        Returns
        -------
        final_error : `float`
            The final error at the end of the fitting procedure.
        """
        if self.gt_shape is not None:
            return compute_error(self.final_shape, self.gt_shape, error_type)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def initial_error(self, error_type='me_norm'):
        r"""
        Returns the initial fitting error.

        Parameters
        -----------
        error_type : `str` ``{'me_norm', 'me', 'rmse'}``, optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

        Returns
        -------
        initial_error : `float`
            The initial error at the start of the fitting procedure.
        """
        if self.gt_shape is not None:
            return compute_error(self.initial_shape, self.gt_shape, error_type)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def plot_errors(self, error_type='me_norm', figure_id=None,
                    new_figure=False, render_lines=True, line_colour='b',
                    line_style='-', line_width=2, render_markers=True,
                    marker_style='o', marker_size=4, marker_face_colour='b',
                    marker_edge_colour='k', marker_edge_width=1.,
                    render_axes=True, axes_font_name='sans-serif',
                    axes_font_size=10, axes_font_style='normal',
                    axes_font_weight='normal', figure_size=(10, 6),
                    render_grid=True, grid_line_style='--',
                    grid_line_width=0.5):
        r"""
        Plot of the error evolution at each fitting iteration.
        Parameters
        ----------
        error_type : {``me_norm``, ``me``, ``rmse``}, optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``} or
                      ``(3, )`` `ndarray`, optional
            The colour of the lines.
        line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : {``.``, ``,``, ``o``, ``v``, ``^``, ``<``, ``>``, ``+``,
                        ``x``, ``D``, ``d``, ``s``, ``p``, ``*``, ``h``, ``H``,
                        ``1``, ``2``, ``3``, ``4``, ``8``}, optional
            The style of the markers.
        marker_size : `int`, optional
            The size of the markers in points^2.
        marker_face_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                             or ``(3, )`` `ndarray`, optional
            The face (filling) colour of the markers.
        marker_edge_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                             or ``(3, )`` `ndarray`, optional
            The edge colour of the markers.
        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : {``serif``, ``sans-serif``, ``cursive``, ``fantasy``,
                          ``monospace``}, optional
            The font of the axes.
        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : {``ultralight``, ``light``, ``normal``, ``regular``,
                            ``book``, ``medium``, ``roman``, ``semibold``,
                            ``demibold``, ``demi``, ``bold``, ``heavy``,
                            ``extra bold``, ``black``}, optional
            The font weight of the axes.
        figure_size : (`float`, `float`) or `None`, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.
        Returns
        -------
        viewer : :map:`GraphPlotter`
            The viewer object.
        """
        from menpo.visualize import GraphPlotter
        errors_list = self.errors(error_type=error_type)
        return GraphPlotter(figure_id=figure_id, new_figure=new_figure,
                            x_axis=range(len(errors_list)),
                            y_axis=[errors_list],
                            title='Fitting Errors per Iteration',
                            x_label='Iteration', y_label='Fitting Error',
                            x_axis_limits=(0, len(errors_list)-1),
                            y_axis_limits=None).render(
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, render_grid=render_grid,
            grid_line_style=grid_line_style, grid_line_width=grid_line_width,
            figure_size=figure_size)

    def displacements(self):
        r"""
        A list containing the displacement between the shape of each iteration
        and the shape of the previous one.
        :type: `list` of ndarray
        """
        return [np.linalg.norm(s1.points - s2.points, axis=1)
                for s1, s2 in zip(self.shapes, self.shapes[1:])]

    def displacements_stats(self, stat_type='mean'):
        r"""
        A list containing the a statistical metric on the displacement between
        the shape of each iteration and the shape of the previous one.
        Parameters
        -----------
        stat_type : `str` ``{'mean', 'median', 'min', 'max'}``, optional
            Specifies a statistic metric to be extracted from the displacements.
        Returns
        -------
        :type: `list` of `float`
            The statistical metric on the points displacements for each
            iteration.
        """
        if stat_type == 'mean':
            return [np.mean(d) for d in self.displacements()]
        elif stat_type == 'median':
            return [np.median(d) for d in self.displacements()]
        elif stat_type == 'max':
            return [np.max(d) for d in self.displacements()]
        elif stat_type == 'min':
            return [np.min(d) for d in self.displacements()]
        else:
            raise ValueError("type must be 'mean', 'median', 'min' or 'max'")

    def plot_displacements(self, stat_type='mean', figure_id=None,
                           new_figure=False, render_lines=True, line_colour='b',
                           line_style='-', line_width=2, render_markers=True,
                           marker_style='o', marker_size=4,
                           marker_face_colour='b', marker_edge_colour='k',
                           marker_edge_width=1., render_axes=True,
                           axes_font_name='sans-serif', axes_font_size=10,
                           axes_font_style='normal', axes_font_weight='normal',
                           figure_size=(10, 6), render_grid=True,
                           grid_line_style='--', grid_line_width=0.5):
        r"""
        Plot of a statistical metric of the displacement between the shape of
        each iteration and the shape of the previous one.
        Parameters
        ----------
        stat_type : {``mean``, ``median``, ``min``, ``max``}, optional
            Specifies a statistic metric to be extracted from the displacements
            (see also `displacements_stats()` method).
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``} or
                      ``(3, )`` `ndarray`, optional
            The colour of the lines.
        line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : {``.``, ``,``, ``o``, ``v``, ``^``, ``<``, ``>``, ``+``,
                        ``x``, ``D``, ``d``, ``s``, ``p``, ``*``, ``h``, ``H``,
                        ``1``, ``2``, ``3``, ``4``, ``8``}, optional
            The style of the markers.
        marker_size : `int`, optional
            The size of the markers in points^2.
        marker_face_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                             or ``(3, )`` `ndarray`, optional
            The face (filling) colour of the markers.
        marker_edge_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                             or ``(3, )`` `ndarray`, optional
            The edge colour of the markers.
        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : {``serif``, ``sans-serif``, ``cursive``, ``fantasy``,
                          ``monospace``}, optional
            The font of the axes.
        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : {``ultralight``, ``light``, ``normal``, ``regular``,
                            ``book``, ``medium``, ``roman``, ``semibold``,
                            ``demibold``, ``demi``, ``bold``, ``heavy``,
                            ``extra bold``, ``black``}, optional
            The font weight of the axes.
        figure_size : (`float`, `float`) or `None`, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.
        Returns
        -------
        viewer : :map:`GraphPlotter`
            The viewer object.
        """
        from menpo.visualize import GraphPlotter
        # set labels
        if stat_type == 'max':
            ylabel = 'Maximum Displacement'
            title = 'Maximum displacement per Iteration'
        elif stat_type == 'min':
            ylabel = 'Minimum Displacement'
            title = 'Minimum displacement per Iteration'
        elif stat_type == 'mean':
            ylabel = 'Mean Displacement'
            title = 'Mean displacement per Iteration'
        elif stat_type == 'median':
            ylabel = 'Median Displacement'
            title = 'Median displacement per Iteration'
        else:
            raise ValueError('stat_type must be one of {max, min, mean, '
                             'median}.')
        # plot
        displacements_list = self.displacements_stats(stat_type=stat_type)
        return GraphPlotter(figure_id=figure_id, new_figure=new_figure,
                            x_axis=range(len(displacements_list)),
                            y_axis=[displacements_list],
                            title=title,
                            x_label='Iteration', y_label=ylabel,
                            x_axis_limits=(0, len(displacements_list)-1),
                            y_axis_limits=None).render(
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, render_grid=render_grid,
            grid_line_style=grid_line_style, grid_line_width=grid_line_width,
            figure_size=figure_size)

    def as_serializableresult(self):
        return SerializableResult(self.image, self.shapes(), self.n_iters,
                                  gt_shape=self.gt_shape)

    def __str__(self):
        out = "Initial error: {0:.4f}\nFinal error: {1:.4f}".format(
            self.initial_error(), self.final_error())
        return out


class AlgorithmResult(Result):

    @property
    def n_iters(self):
        return len(self.shapes()) - 1

    @property
    def transforms(self):
        r"""
        Generates a list containing the transforms obtained at each fitting
        iteration.
        """
        return [self.fitter.transform.from_vector(p)
                for p in self.shape_parameters]

    @property
    def final_transform(self):
        r"""
        Returns the final transform.
        """
        return self.fitter.transform.from_vector(self.shape_parameters[-1])

    @property
    def initial_transform(self):
        r"""
        Returns the initial transform from which the fitting started.
        """
        return self.fitter.transform.from_vector(self.shape_parameters[0])

    def shapes(self, as_points=False):
        if as_points:
            return [self.fitter.transform.from_vector(p).target.points
                    for p in self.shape_parameters]

        else:
            return [self.fitter.transform.from_vector(p).target
                    for p in self.shape_parameters]

    @property
    def final_shape(self):
        return self.final_transform.target

    @property
    def initial_shape(self):
        return self.initial_transform.target


class FitterResult(Result):

    def __init__(self, image, fitter, algorithm_results, affine_correction,
                 gt_shape=None):
        super(FitterResult, self).__init__()
        self.image = image
        self.fitter = fitter
        self.algorithm_results = algorithm_results
        self._affine_correction = affine_correction
        self._gt_shape = gt_shape

    @property
    def n_levels(self):
        r"""
        The number of levels of the fitter object.

        :type: `int`
        """
        return self.fitter.n_levels

    @property
    def scales(self):
        return self.fitter.scales

    @property
    def n_iters(self):
        r"""
        The total number of iterations used to fitter the image.

        :type: `int`
        """
        n_iters = 0
        for f in self.algorithm_results:
            n_iters += f.n_iters
        return n_iters

    def shapes(self, as_points=False):
        r"""
        Generates a list containing the shapes obtained at each fitting
        iteration.

        Parameters
        -----------
        as_points : `boolean`, optional
            Whether the result is returned as a `list` of :map:`PointCloud` or
            a `list` of `ndarrays`.

        Returns
        -------
        shapes : `list` of :map:`PointCoulds` or `list` of `ndarray`
            A list containing the fitted shapes at each iteration of
            the fitting procedure.
        """
        shapes = []
        for j, (alg, s) in enumerate(zip(self.algorithm_results, self.scales)):
            transform = Scale(self.scales[-1]/s, alg.final_shape.n_dims)
            for t in alg.shapes(as_points=as_points):
                t = transform.apply(t)
                shapes.append(self._affine_correction.apply(t))

        return shapes

    @property
    def final_shape(self):
        r"""
        The final fitted shape.

        :type: :map:`PointCloud`
        """
        final_shape = self.algorithm_results[-1].final_shape
        return self._affine_correction.apply(final_shape)

    @property
    def initial_shape(self):
        initial_shape = self.algorithm_results[0].initial_shape
        Scale(self.scales[-1]/self.scales[0],
              initial_shape.n_dims).apply_inplace(initial_shape)
        return self._affine_correction.apply(initial_shape)


class SerializableResult(Result):

    def __init__(self, image, shapes, n_iters, gt_shape=None):
        self.image = image
        self._gt_shape = gt_shape
        self._gt_shape = gt_shape
        self._shapes = shapes
        self._n_iters = n_iters

    @property
    def n_iters(self):
        return self._n_iters

    def shapes(self, as_points=False):
        if as_points:
            return [s.points for s in self._shapes]
        else:
            return self._shapes

    @property
    def initial_shape(self):
        return self._shapes[0]

    @property
    def final_shape(self):
        return self._shapes[-1]
