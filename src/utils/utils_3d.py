"""
    :filename utils_3d.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    This file contains functions needed to cover the method part that occurs within the 3D scene.
"""

import vtk
import numpy as np

from scipy.spatial import distance


def get_distance_meter_color(distance):
    """
    Function transforms the normalized distance of two points into a RGB color.
    This color is interpolated between Green (0, 255, 0) through Yellow (255, 255, 0) to Red (255, 0, 0).

    :param distance: Normalized distance between GT and prediction, in range (0, 1).

    :return: (R, G, B) representation of the distance.
    """

    R, G, B = 0.0, 0.0, 0.0

    '''From Green to Yellow for the first half.'''
    if 0 <= distance < 0.5:
        G = 1.0
        R = 2 * distance

    '''From Yellow to Red in the second half.'''
    if 0.5 <= distance <= 1:
        R = 1.0
        G = 1.0 - 2 * (distance - 0.5)

    return R, G, B


def plot_ray_in_scene(p1, p2, renderer):
    """
    Function adds a line in the 3D scene.
    It is necessary to provide the starting and end point of such line, as well as
    the renderer to which the actor should be added.

    :param p1: Line starting point in 3-space. Form: tuple, (x, y, z).
    :param p2: Line end point in 3-space. Form: tuple, (x, y, z).
    :param renderer: Renderer object instance to which the line's actor will be attached.
    """

    if p1 is None or p2 is None:
        return

    '''VTK point objects preparation.'''
    points = vtk.vtkPoints()
    points.InsertNextPoint(p1)
    points.InsertNextPoint(p2)

    '''VTK line preparation.'''
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, 0)
    line.GetPointIds().SetId(1, 1)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(cells)

    '''Setup actor and mapper.'''
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polyData)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d('Tomato'))

    renderer.AddActor(actor)


def plot_landmark_in_scene(position, renderer, color='Tomato', radius=0.8, dist_from_gt=None):
    """
    Function creates a sphere representing landmark on jaw model.
    Afterwards sphere is mapped on its actor and actor is added to renderer.

    :param position: Center of the landmark in 3-space. Form: tuple (x, y, z).
    :param renderer: Renderer object instance to which the sphere's actor will be attached.
    :param color: String value of color of the sphere. Check VTK documentation for possibilities.
    :param radius: Radius of sphere in mm.
    :param dist_from_gt: Distance from GT.
    """

    '''Sphere object declaration, positioning and radius setup.'''
    landmark = vtk.vtkSphereSource()
    landmark.SetCenter(position)
    landmark.SetRadius(radius)

    '''Make the surface smooth.'''
    landmark.SetPhiResolution(100)
    landmark.SetThetaResolution(100)
    landmark.Update()

    '''Setup actor and mapper.'''
    landmark_mapper = vtk.vtkPolyDataMapper()
    landmark_mapper.SetInputConnection(landmark.GetOutputPort())
    landmark_actor = vtk.vtkActor()
    landmark_actor.SetMapper(landmark_mapper)

    if dist_from_gt is None:
        landmark_actor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d(color))
    else:
        if dist_from_gt >= 10.0:
            normalized_dist = 1.0
        else:
            normalized_dist = dist_from_gt / 10.0

        r, g, b = get_distance_meter_color(normalized_dist)

        landmark_actor.GetProperty().SetColor(r, g, b)

    renderer.AddActor(landmark_actor)


def display_text_in_scene(text, position, renderer, color='Tomato', dist_from_gt=None):
    """
    Function adds text in the scene.
    Text is in form of vtkVectorText and it's actor is attached to provided renderer.

    :param text: The string value of text to be displayed in the scene.
    :param position: The (x, y, z) position in 3-space of the text.
    :param renderer: Renderer object instance to which the text's actor will be attached.
    :param color: String value of color of the text. Check VTK documentation for possibilities.
    :param dist_from_gt: Distance of corresponding landmark from the GT
    """

    '''vtkVectorText creation and text setting.'''
    atext = vtk.vtkVectorText()
    atext.SetText(text)

    '''Setup actor and mapper.'''
    textMapper = vtk.vtkPolyDataMapper()
    textMapper.SetInputConnection(atext.GetOutputPort())
    textActor = vtk.vtkFollower()
    textActor.SetMapper(textMapper)

    if dist_from_gt is None:
        textActor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d(color))
    else:
        if dist_from_gt >= 10.0:
            normalized_dist = 1.0
        else:
            normalized_dist = dist_from_gt / 10.0

        r, g, b = get_distance_meter_color(normalized_dist)

        textActor.GetProperty().SetColor(r, g, b)

    '''Text scaling and positioning.'''
    textActor.SetScale(0.8, 0.8, 0.8)
    textActor.AddPosition(position[0], position[1], position[2])

    renderer.AddActor(textActor)
    textActor.SetCamera(renderer.GetActiveCamera())


def find_closest_point_on_mesh(point, mesh, transformation):
    """
    Function finds closest point located on given mesh surface.

    :param point: An original point. Its closest location to given polygon is calculated. Form: (x, y, z).
    :param mesh: Polygonal data of a mesh.
    :param transformation: Transformation applied on polygon model. As the Cell Locator does not take into account
    applied transformations, it is necessary to reverse the estimated point to original value first.

    :return: (x, y, z) coordinates on polygon.
    """

    res = np.zeros(3)

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    '''Private properties to be filled by locator method.'''
    cell_id = vtk.mutable(0)
    sub_id = vtk.mutable(0)
    dist = vtk.reference(0.0)

    '''Firstly, transform the predicted point into original position.'''
    point = transform_landmark(point, transformation)

    '''Calculate point location on polygon in original space.'''
    locator.FindClosestPoint(point, res, cell_id, sub_id, dist)

    '''Transform landmark back into its estimated position.'''
    res = transform_landmark(res, transformation.GetInverse())

    return res


def transform_landmark(landmark, t):
    """
    Function applies transformation t to a single landmark represented by its coordinates in space.
    It first creates poly data from its positions and then applies given transformation.

    :param landmark: (x, y, z) coordinates of landmark to be transformed
    :param t: transformation to be applied, form: vtkTransform()

    :return: (x, y, z) coordinates of landmark after transformation
    """

    '''VTK Points object preparation.'''
    points = vtk.vtkPoints()
    pd = vtk.vtkPolyData()
    pid = points.InsertNextPoint(landmark)
    pd.SetPoints(points)

    '''Getting the inverse transformation to acquire the original position.'''
    trans = vtk.vtkTransformPolyDataFilter()
    trans.SetInputData(pd)
    trans.SetTransform(t.GetInverse())
    trans.Update()
    pd_trans = trans.GetOutput()

    p = None
    for lm_no in range(pd_trans.GetNumberOfPoints()):
        p = pd_trans.GetPoint(lm_no)

    return p[0], p[1], p[2]


def get_distance_two_points_3d(p1, p2):
    """
    Find the Euclidean distance between two points in 3-space.

    :param p1: First point, tuple (x, y, z)
    :param p2: Second point, tuple (x, y, z)

    :return: Euclidean distance between point p1 and p2.
    """

    return distance.euclidean(p1, p2)
