"""
    :filename Evaluator.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    Evaluator class - the core of the evaluation of unseen polygon meshes.
"""

import vtk
import math
import torch
import datetime
import numpy as np
import vtk.util.numpy_support

from torch.utils.data.dataloader import DataLoader

import config

from src.JawModel import get_new_model
from src.trainer.JawDataset import EvalJawDataset
from src.utils.utils_2d import non_maxima_suppression
from src.utils.utils_3d import find_closest_point_on_mesh, get_distance_two_points_3d, plot_landmark_in_scene, display_text_in_scene
from src.evaluator.ConsensusCENTROID import ConsensusCENTROID
from src.evaluator.GeometricConsensusRANSAC import GeometricConsensusRANSAC


class Evaluator:
    class KeyInteractor(vtk.vtkInteractorStyleTrackballCamera):
        """
        Interactor class for custom key events - evaluation start, next model loading.
        """

        def __init__(self, interactor, evaluator, *args, **kwargs):

            super().__init__(*args, **kwargs)

            self.Parent = interactor
            self.evaluator = evaluator
            self.AddObserver(vtk.vtkCommand.KeyPressEvent, self.keypress_callback_function)

        def keypress_callback_function(self, _obj, _event):
            key = self.Parent.GetKeySym()

            if key == 'n':
                '''Remove annotated landmarks.'''
                self.evaluator.renderer.RemoveAllViewProps()
                self.evaluator.renderer.AddActor(self.evaluator.jaw.actor)

            if key == 'g':
                '''Evaluate and load next model.'''
                self.evaluator.net_inputs_renders = []
                self.evaluator.get_NN_inputs()
                self.evaluator.dataset = EvalJawDataset(net_input_renders=self.evaluator.net_inputs_renders,
                                                        input_format=self.evaluator.input_format)
                self.evaluator.eval_loader = DataLoader(self.evaluator.dataset, batch_size=1, shuffle=False,
                                                        pin_memory=True)

                self.evaluator.pred_pos_tmp = None

                self.evaluator.evaluate()
                self.evaluator.window.Render()

            if key == 'Tab':
                '''Skip loaded mesh.'''
                if not self.evaluator.single:
                    self.evaluator.model_idx += 1
                    self.evaluator.renderer.RemoveAllViewProps()
                    self.evaluator.jaw = get_new_model(self.evaluator.curr_model_path)
                    self.evaluator.transform = self.evaluator.jaw.apply_jaw_transformations()

                    self.evaluator.renderer.AddActor(self.evaluator.jaw.actor)

                    self.evaluator.renderer.ResetCamera()
                    self.evaluator.renderer.GetActiveCamera().OrthogonalizeViewUp()
                    self.evaluator.renderer.GetActiveCamera().SetViewUp(0.0, 1.0, 0.0)

                    self.evaluator.camera = self.evaluator.renderer.GetActiveCamera()

                    self.evaluator.window.Render()

                    win_name = 'Evaluating ' + str(self.evaluator.jaw.name)
                    self.evaluator.window.SetWindowName(win_name)

                    try:
                        self.evaluator.curr_model_path = next(self.evaluator.model_paths)
                    except StopIteration:
                        self.evaluator.finish_evaluating()  # No more models to evaluator

    def __init__(self, network, model_path, perf_meas=None, single=False, centroid=False, input_format='depth'):
        self.network = network
        self.model_paths = iter(model_path)
        self.perf_meas = perf_meas
        self.model_idx = 0
        self.single = single
        self.centroid = centroid  # Calculate and measure centroid consensus method if set to True
        self.input_format = input_format

        if self.single:
            self.curr_model_path = model_path
        else:
            self.curr_model_path = next(self.model_paths)

        self.jaw = get_new_model(self.curr_model_path)  # All necessary data for rendering are prepared

        self.camera = None
        self.window = None
        self.renderer = None
        self.interactor = None

        self.picker = vtk.vtkCellPicker()
        self.picker.AddObserver("EndPickEvent", self.poly_intersect)

        self.transform = self.jaw.apply_jaw_transformations()
        self.initialise_scene()

        self.net_inputs_renders = []

        self.distances_mean = []
        self.distances_RANSAC = []
        self.eval_loader = None
        self.pred_pos_tmp = None

        self.ot_eval = 0
        self.ot_ransac = 0

        self.certainties_vals = [[] for _ in range(config.LANDMARKS_NUM)]

    def evaluate(self):
        """Evaluation method. Evaluation starts here."""

        origins = [[] for _ in range(config.LANDMARKS_NUM)]
        line_ends = [[] for _ in range(config.LANDMARKS_NUM)]

        heatmaps_max_vals = dict()

        missing_lds = []

        for GT in self.jaw.GTs:
            if GT[0] and GT[1] and GT[2] == math.inf:
                missing_lds.append(int(GT[3]))

        start_time = datetime.datetime.now()

        for i_batch, sample in enumerate(self.eval_loader):
            image = sample['image'].type(torch.FloatTensor)
            image = image.to('cuda:0' if torch.cuda.is_available() else 'cpu')

            '''Output of network is 32 heatmaps, thus prediction is 32x128x128.'''
            prediction = self.network(image)

            origins_tmp = []
            line_ends_tmp = []

            valid_missing_preds = 0

            '''For each heatmap of prediction:'''
            for i in range(0, config.LANDMARKS_NUM):
                '''Get the (x, y) display coordinates.'''
                x, y, is_present_pred = non_maxima_suppression(prediction.squeeze(0)[i],
                                                               self.perf_meas,
                                                               i,
                                                               heatmaps_max_vals)

                if not self.single:
                    self.perf_meas.heatmap_certainties_preds.append([is_present_pred,
                                                                     config.TOOTH_TO_CSV_NOTATION[
                                                                         i + 1] not in missing_lds,
                                                                     i + 1])

                if is_present_pred is (config.TOOTH_TO_CSV_NOTATION[i + 1] not in missing_lds):
                    valid_missing_preds += 1

                coeff = config.DIMENSIONS['original'] / config.DIMENSIONS['output_net']

                '''Set camera to the same position as when the depth map was rendered.'''
                self.camera.SetPosition(sample['camera'][0], sample['camera'][1], sample['camera'][2])

                if self.centroid or config.MULTI_VIEW_NUM == 1:
                    self.window.Render()
                    _res = self.picker.Pick(round(coeff * y, 3), round(coeff * x, 3), 0, self.renderer)

                coordinate = vtk.vtkCoordinate()
                coordinate.SetCoordinateSystemToDisplay()
                coordinate.SetValue(coeff * y, coeff * x)

                '''Convert the display coordinates into World coordinates.'''
                display_coordinates = coordinate.GetComputedWorldValue(self.renderer)

                origins[i].append(
                    [sample['camera'][0].numpy()[0], sample['camera'][1].numpy()[0], sample['camera'][2].numpy()[0]])

                if self.centroid or config.MULTI_VIEW_NUM == 1:
                    if self.pred_pos_tmp is not None:
                        line_ends[i].append([self.pred_pos_tmp[0],
                                             self.pred_pos_tmp[1],
                                             self.pred_pos_tmp[2]])
                else:
                    line_ends[i].append([display_coordinates[0],
                                         display_coordinates[1],
                                         display_coordinates[2]])

                origins.append([origins_tmp])
                line_ends.append([line_ends_tmp])

        end_time = datetime.datetime.now()
        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds() * 1000
        self.ot_eval += execution_time
        print('Evaluation computational time: {:.0f} ms'.format(execution_time))

        '''For each predicted value:'''
        ransac_time = 0

        for ld_id, i in zip(config.VALID_NOTATIONS, range(0, config.LANDMARKS_NUM)):
            if ld_id not in missing_lds:
                origins_tmp = np.array(origins[i])
                line_ends_tmp = np.array(line_ends[i])

                '''Calculate estimation by Centroid and find closest point on surface of jaw.'''
                if self.centroid:
                    centroid_estimator = ConsensusCENTROID(line_ends_tmp)
                    est_mean = centroid_estimator.estimate_CENTROID()
                    est_mean_on_poly = find_closest_point_on_mesh(est_mean, self.jaw.poly, self.transform)
                    dist_GT_mean = get_distance_two_points_3d(est_mean_on_poly, self.jaw.GTs[i][:3])

                    plot_landmark_in_scene(est_mean_on_poly, self.renderer, 'Purple', dist_from_gt=dist_GT_mean)
                    display_text_in_scene(text=str(config.LM_TO_TECH_REPORT_NOTATION[i + 1]),
                                          position=(est_mean_on_poly[0] + 0.6,
                                                    est_mean_on_poly[1] + 0.6,
                                                    est_mean_on_poly[2] + 0.6),
                                          renderer=self.renderer,
                                          color='Purple',
                                          dist_from_gt=dist_GT_mean)

                    if self.perf_meas is not None:
                        self.perf_meas.append_CENTROID(self.model_idx,
                                                       i,
                                                       round(get_distance_two_points_3d(self.jaw.GTs[i][:3],
                                                                                        est_mean_on_poly), 3))

                '''Calculate estimation by RANSAC algorithm and find closest point on surface of jaw.'''
                start_time = datetime.datetime.now()

                if len(line_ends[i]) != 1:
                    ransac_estimator = GeometricConsensusRANSAC(origins_tmp, line_ends_tmp)
                    est_ransac = ransac_estimator.estimate_RANSAC()
                    est_ransac_on_poly = find_closest_point_on_mesh(est_ransac, self.jaw.poly, self.transform)
                else:
                    '''One view.'''
                    est_ransac_on_poly = find_closest_point_on_mesh(
                        (line_ends[i][0][0], line_ends[i][0][1], line_ends[i][0][2]), self.jaw.poly, self.transform)

                if i % 2 == 0:
                    temp_pos = est_ransac_on_poly
                    temp_ransac_inliers = ransac_estimator.pred_inliers

                end_time = datetime.datetime.now()
                time_diff = (end_time - start_time)
                execution_time = time_diff.total_seconds() * 1000
                ransac_time += execution_time

                if i % 2 == 1:
                    certain_i_1 = sum(j > config.BIN_CLASSIFIER_THRESHOLD for j in heatmaps_max_vals[i - 1])
                    certain_i = sum(j > config.BIN_CLASSIFIER_THRESHOLD for j in heatmaps_max_vals[i])

                    total_certain_preds = certain_i + certain_i_1

                    if total_certain_preds != 0:
                        relative_percent_from_RANSAC = (ransac_estimator.pred_inliers + temp_ransac_inliers) / total_certain_preds
                    # print(f'---- Relative RANSAC certainty: {relative_percent_from_RANSAC}')

                    if certain_i + certain_i_1 >= 2 * config.PARTIAL_PREDS_THRESHOLD or (
                            total_certain_preds > 18 and relative_percent_from_RANSAC >= .85):
                        if ld_id - 1 not in missing_lds:
                            dist_GT_ransac = get_distance_two_points_3d(temp_pos, self.jaw.GTs[i-1][:3])

                            plot_landmark_in_scene(temp_pos, self.renderer, 'Blue', dist_from_gt=dist_GT_ransac)
                            display_text_in_scene(text=str(config.SCENE_NAMES[i]),
                                                  position=(temp_pos[0] + 0.6, temp_pos[1] + 0.6,
                                                            temp_pos[2] + 0.6),
                                                  renderer=self.renderer,
                                                  color='Blue',
                                                  dist_from_gt=dist_GT_ransac)

                            self.certainties_vals[i - 1].append(['TP'])

                            if self.perf_meas is not None:
                                self.perf_meas.TP_3D += 1
                                self.perf_meas.append_RANSAC(self.model_idx, i - 1, round(
                                    get_distance_two_points_3d(self.jaw.GTs[i - 1][:3], temp_pos), 3))
                        # FP
                        else:
                            plot_landmark_in_scene(temp_pos, self.renderer, 'Black')
                            display_text_in_scene(text=str(config.SCENE_NAMES[i]),
                                                  position=(temp_pos[0] + 0.6, temp_pos[1] + 0.6,
                                                            temp_pos[2] + 0.6),
                                                  renderer=self.renderer,
                                                  color='Black')
                            self.certainties_vals[i - 1].append(['FP'])

                            if self.perf_meas is not None:
                                self.perf_meas.FP_3D += 1

                        # TP
                        if ld_id not in missing_lds:
                            dist_GT_ransac = get_distance_two_points_3d(est_ransac_on_poly, self.jaw.GTs[i][:3])

                            plot_landmark_in_scene(est_ransac_on_poly, self.renderer, 'Blue', dist_from_gt=dist_GT_ransac)
                            display_text_in_scene(text=str(config.SCENE_NAMES[i + 1]),
                                                  position=(est_ransac_on_poly[0] + 0.6, est_ransac_on_poly[1] + 0.6,
                                                            est_ransac_on_poly[2] + 0.6),
                                                  renderer=self.renderer,
                                                  color='Blue',
                                                  dist_from_gt=dist_GT_ransac)
                            self.certainties_vals[i].append(['TP'])

                            if self.perf_meas is not None:
                                self.perf_meas.TP_3D += 1
                                self.perf_meas.append_RANSAC(self.model_idx, i, round(
                                    get_distance_two_points_3d(self.jaw.GTs[i][:3], est_ransac_on_poly), 3))
                        # FP
                        else:
                            plot_landmark_in_scene(est_ransac_on_poly, self.renderer, 'Black')
                            display_text_in_scene(text=str(config.LM_TO_TECH_REPORT_NOTATION[i + 1]),
                                                  position=(est_ransac_on_poly[0] + 0.6, est_ransac_on_poly[1] + 0.6,
                                                            est_ransac_on_poly[2] + 0.6),
                                                  renderer=self.renderer,
                                                  color='Black')
                            self.certainties_vals[i].append(['FP'])

                            if self.perf_meas is not None:
                                self.perf_meas.FP_3D += 1

                    else:
                        if ld_id - 1 not in missing_lds:
                            self.certainties_vals[i - 1].append(['FN'])

                            if self.perf_meas is not None:
                                self.perf_meas.FN_3D += 1
                        # TN
                        else:
                            self.certainties_vals[i - 1].append(['TN'])

                            if self.perf_meas is not None:
                                self.perf_meas.TN_3D += 1

                        # FN
                        if ld_id not in missing_lds:
                            self.certainties_vals[i].append(['FN'])

                            if self.perf_meas is not None:
                                self.perf_meas.FN_3D += 1
                        # TN
                        else:
                            self.certainties_vals[i].append(['TN'])

                            if self.perf_meas is not None:
                                self.perf_meas.TN_3D += 1
            else:
                if i % 2 == 1:
                    if ld_id - 1 in missing_lds:
                        self.certainties_vals[i - 1].append(['TN'])
                        if self.perf_meas is not None:
                            self.perf_meas.TN_3D += 1
                    # TN
                    else:
                        self.certainties_vals[i - 1].append(['FN'])

                        if self.perf_meas is not None:
                            self.perf_meas.FN_3D += 1

                    if ld_id in missing_lds:
                        self.certainties_vals[i].append(['TN'])
                        if self.perf_meas is not None:
                            self.perf_meas.TN_3D += 1
                    # TN
                    else:
                        self.certainties_vals[i].append(['FN'])

                        if self.perf_meas is not None:
                            self.perf_meas.FN_3D += 1

        print('RANSAC computational time: {:.0f} ms\n'.format(ransac_time))
        self.ot_ransac += ransac_time

        '''Plot Ground Truths in scene. Uncomment if needed.'''
        for GT in self.jaw.GTs:
            if GT[-1] not in missing_lds:
                plot_landmark_in_scene(GT[:3], self.renderer, color='chocolate')

    def initialise_scene(self):
        """Prepares everything necessary for the mesh and prediction rendering."""

        self.renderer = vtk.vtkRenderer()
        self.window = vtk.vtkRenderWindow()
        self.window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)

        style = self.KeyInteractor(self.interactor, self)
        style.SetDefaultRenderer(self.renderer)
        self.interactor.SetInteractorStyle(style)

        self.renderer.AddActor(self.jaw.actor)
        self.renderer.SetBackground(0, 0, 0)

        self.window.SetSize(config.DIMENSIONS['original'], config.DIMENSIONS['original'])

        self.renderer.ResetCamera()
        self.camera = self.renderer.GetActiveCamera()

        win_name = 'Evaluating ' + str(self.jaw.name)
        self.window.SetWindowName(win_name)

        self.interactor.Initialize()

        self.interactor.SetPicker(self.picker)

    def get_NN_inputs(self):
        """Obtain depth maps for the multi-view approach from different camera positions and store them."""

        assert config.MULTI_VIEW_NUM in [1, 9, 16, 25, 100]

        if config.MULTI_VIEW_NUM == 1:
            camera_initial, range_views, camera_step = 0, 1, 0
        elif config.MULTI_VIEW_NUM == 9:
            camera_initial, range_views, camera_step = 15, 3, 10
        elif config.MULTI_VIEW_NUM == 16:
            camera_initial, range_views, camera_step = 20, 4, 10
        elif config.MULTI_VIEW_NUM == 25:
            camera_initial, range_views, camera_step = 30, 5, 12
        else:
            camera_initial, range_views, camera_step = 30, 10, 6

        self.camera.Azimuth(-camera_initial)
        self.camera.Elevation(-camera_initial)

        for j in range(0, range_views):
            self.camera.Elevation(camera_step)

            if j != 0:
                self.camera.Azimuth(-camera_initial * 2)

            for i in range(0, range_views):
                self.camera.Azimuth(camera_step)

                depth_map_arr = None
                geometry_arr = None

                if self.input_format == 'depth' or self.input_format == 'depth+geom':
                    z_buffer = vtk.vtkWindowToImageFilter()
                    z_buffer.SetInput(self.window)
                    z_buffer.ReadFrontBufferOff()
                    z_buffer.SetInputBufferTypeToZBuffer()
                    z_buffer.Update()

                    z_buffer_vtk_image = z_buffer.GetOutput()
                    width, height, _ = z_buffer_vtk_image.GetDimensions()
                    vtk_array = z_buffer_vtk_image.GetPointData().GetScalars()
                    # components = vtk_array.GetNumberOfComponents()

                    # z_buffer_arr = vtk.util.numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, components)

                    depth_map = vtk.vtkImageShiftScale()
                    depth_map.SetOutputScalarTypeToUnsignedChar()
                    depth_map.SetInputConnection(z_buffer.GetOutputPort())
                    depth_map.SetShift(0)
                    depth_map.SetScale(-255)
                    depth_map.Update()

                    depth_map_vtk_image = depth_map.GetOutput()
                    width, height, _ = depth_map_vtk_image.GetDimensions()
                    vtk_array = depth_map_vtk_image.GetPointData().GetScalars()
                    components = vtk_array.GetNumberOfComponents()

                    depth_map_arr = vtk.util.numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, components)

                if self.input_format == 'geom' or self.input_format == 'depth+geom':
                    grabber = vtk.vtkWindowToImageFilter()
                    grabber.SetInput(self.window)
                    grabber.Update()

                    geom_vtk_image = grabber.GetOutput()
                    width, height, _ = geom_vtk_image.GetDimensions()
                    vtk_array = geom_vtk_image.GetPointData().GetScalars()
                    components = vtk_array.GetNumberOfComponents()

                    geometry_arr = vtk.util.numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, components)

                self.net_inputs_renders.append({'depth_map': depth_map_arr,
                                                'geometry': geometry_arr,
                                                'camera': [self.camera.GetPosition()[0],
                                                           self.camera.GetPosition()[1],
                                                           self.camera.GetPosition()[2]]})

    def poly_intersect(self, _object, _event):
        """
        Picker callback, this is needed for Centroid consensus method as each prediction is predicted on the
        polygonal model's surface first.

        Not used when RANSAC only is evaluated (recommended).
        """

        if self.picker.GetCellId() >= 0:
            pick_pos = self.picker.GetPickPosition()
            self.pred_pos_tmp = pick_pos

    def finish_evaluating(self):
        """Prints all statistics, plots graphs..."""
        print(f'Overall time evaluation: {self.ot_eval} ms')
        print(f'Overall time ransac: {self.ot_ransac} ms')

        '''Get the TP, FP, TN, FN values.'''
        self.perf_meas.certainty_metrics_calc()

        print(f'\n\nPerformance Measure results:\n')

        print(f'TP: {self.perf_meas.TP}, FP: {self.perf_meas.FP}, FN: {self.perf_meas.FN}, TN:{self.perf_meas.TN}.\n')

        print(f'Lm detection Precision: {self.perf_meas.lds_absence_precision}%.')
        print(f'Lm detection Negative Predictive Value: {self.perf_meas.lds_absence_negative_pred_val}%.')
        print(f'Lm detection Accuracy: {self.perf_meas.lds_absence_accuracy}%.')
        print(f'Lm detection Specificity: {self.perf_meas.lds_absence_specificity}%.')
        print(f'Lm detection Sensitivity: {self.perf_meas.lds_absence_sensitivity}%.\n')

        if self.centroid:
            MRE_C = self.perf_meas.mean_radial_error_CENTROID
            print(f'MRE CENTROID: {round(MRE_C, 2)} mm.')

            SD_C = self.perf_meas.SD_CENTROID
            print(f'Standard deviation CENTROID: {round(SD_C, 2)} mm.')

        MRE_R = self.perf_meas.mean_radial_error_RANSAC
        print(f'MRE RANSAC: {round(MRE_R, 2)} mm.\n')

        SD_R = self.perf_meas.SD_RANSAC
        print(f'Standard deviation RANSAC: {round(SD_R, 2)} mm.\n')

        '''Plotting box plots and matching acceptances graphs. Uncomment if needed.'''
        # if self.centroid:
        #   self.perf_meas.get_box_plots_each_landmark(consensus='CENTROID')
        #   self.perf_meas.get_matching_acceptances(consensus='CENTROID')
        # self.perf_meas.get_box_plots_each_landmark()
        # self.perf_meas.get_matching_acceptances()

        if self.centroid:
            SDR_CENTROID_2, SDR_CENTROID_2_5, SDR_CENTROID_4 = self.perf_meas.overall_SDR_CENTROID
            print(
                f'Overall SDR CENTROID: 2 mm: {SDR_CENTROID_2}%, 2.5 mm: {SDR_CENTROID_2_5}%, 4 mm: {SDR_CENTROID_4}%.')

        SDR_RANSAC_2, SDR_RANSAC_2_5, SDR_RANSAC_4 = self.perf_meas.overall_SDR_RANSAC
        print(f'Overall SDR RANSAC: 2 mm: {SDR_RANSAC_2}%, 2.5 mm: {SDR_RANSAC_2_5}%, 4 mm: {SDR_RANSAC_4}%.\n')

        exit(0)
