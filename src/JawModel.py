"""
    :filename JawModel.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    File contains JawModel class creation.
    This class is used to represent the polygonal model of a jaw in scene.
"""

import os
import vtk
import math
import numpy as np
import xml.etree.ElementTree as ElementTree

import config


def expand_affine_transformation_r_c(t_matrix):
    """
    | a b c |       | a b c 0 |
    | d e f |   ->  | d e f 0 |
    | g h i |       | g h i 0 |
                    | 0 0 0 1 |

    :param t_matrix: 4x4 vtkMatrix

    :return: 4x4 matrix expanded by zeros in last row and column,
    except for the lower-right corner which is set to 1.
    """

    t_matrix.SetElement(0, 3, 0.0)
    t_matrix.SetElement(1, 3, 0.0)
    t_matrix.SetElement(2, 3, 0.0)
    t_matrix.SetElement(3, 3, 1.0)
    t_matrix.SetElement(3, 0, 0.0)
    t_matrix.SetElement(3, 1, 0.0)
    t_matrix.SetElement(3, 2, 0.0)

    return t_matrix


def get_new_model(model_path, mode='eval'):
    """
    Function creates an instance of JawModel. This object instance contains all necessary
    information for further processing in the pipeline, namely:
    Model path, model name, information for the rendering (poly data, actor) and in case of
    eval mode, the ground truth positions.

    :param model_path: The absolute or relative path to the STL object file
    :param mode: Which module called this function (eval/annotate)
    :return: instance of JawModel
    """
    reader = read_poly_data(model_path)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    reader.Update()
    poly_data = reader.GetOutput()
    jawActor = vtk.vtkActor()
    jawActor.SetMapper(mapper)

    return JawModel(model_path, jaw_poly=poly_data, jaw_actor=jawActor, mode=mode)


def read_poly_data(file_name):
    """
    Function takes file name(path) as an input and creates a poly model using vtk STL reader.
    Both reader and data are returned as both of them are needed in further pipeline processing.

    :param file_name: Path to the STL file to be read.

    :return: STL Read and the mesh data.
    """

    poly_reader = vtk.vtkSTLReader()
    poly_reader.SetFileName(file_name)
    poly_reader.Update()

    return poly_reader


class JawModel:
    """
    Class represents currently processed model of jaw.
    Jaw is defined by path of source file, its poly and actor.
    It also contains the ground truth labels of landmarks associated with the dentition.
    """

    def __init__(self, jaw_path, jaw_poly, jaw_actor, mode='trainer'):
        """
        Class constructor.
        Instance of JawModel is usually created after connecting jaw poly on its actor.
        """
        self.path = jaw_path
        self.name = self.get_name_from_path()
        self.poly = jaw_poly
        self.actor = jaw_actor
        self.mass_center = self.get_mass_center()  # mass center is used in the transformation matrix creation
        self.transformation_matrix = self.get_transformation_matrix()
        # self.jaw_type = self.get_jaw_type()  # maxillary or mandibular
        self.mode = mode
        self.GTs = self.get_GTs()  # Ground truth landmark values

    def get_mass_center(self):
        """
        Method calculates mass center of given model.
JawModel
        :return: tuple (x, y, z) representing the mass center of a 3D model.
        """

        center_of_mass_filter = vtk.vtkCenterOfMass()
        center_of_mass_filter.SetInputData(self.poly)
        center_of_mass_filter.SetUseScalarsAsWeights(False)
        center_of_mass_filter.Update()

        return center_of_mass_filter.GetCenter()

    def get_name_from_path(self):
        """
        Method is used to parse name of file excluding file type from its path.

        :return: Extracted name from the file path, e.g. 000016_0.
        """

        return os.path.splitext(os.path.basename(self.path))[0]

    def get_transformation_matrix(self):
        """
        Method extracts transformation matrix from corresponding XML file, which should be provided with the polygon.

        :return: 4x4 VTK transformation matrix from the XML file.
        """

        path_to_xml = self.path[0:self.path.rfind('_')] + '.xml'
        model_number = self.path[self.path.rfind('_')+1:self.path.rfind('.')]

        return self.get_transformation_matrix_from_xml(path_to_xml, model_number)

    def get_jaw_type(self):
        """
        Method finds the type of the dentition.
        It searches in provided list of lower jaws the presence of the jaw.

        :return: jaw type: 'upper' or 'lower'.
        """

        try:
            file = open('src/lower-jaw-list.txt', 'r', newline='')
        except FileNotFoundError:
            file = open('lower-jaw-list.txt', 'r', newline='')
        finally:
            lower_jaws = file.readlines()

            for lower_jaw in lower_jaws:
                if lower_jaw.strip() == self.path[self.path.rfind('/')+1:]:

                    return 'lower'

        return 'upper'

    def get_GTs(self):
        """
        Method loads ground truth landmark values from corresponding csv file and saves them in an numpy array.

        :return: None in annotation mode, 32x3 Matrix of (x, y, z) GT values in evaluation mode.
        """

        if self.mode == 'eval':
            try:
                file = open('../test-data/annotations/' + self.name + '.csv')
            except FileNotFoundError:
                file = open('test-data/annotations/' + self.name + '.csv')
            finally:
                values = np.loadtxt(file, delimiter=',', usecols=(0, 1, 2, 3), skiprows=1)

                notations = values[:, -1]
                for notation in config.VALID_NOTATIONS:
                    if notation not in notations:
                        values = np.append(values, [math.inf, math.inf, math.inf, notation])  # missing lm -> INF

                values = np.reshape(values, (-1, 4))  # reshape from flatten array into desired
                values = np.array(sorted(values, key=lambda x: x[-1]))  # sort landmarks according to the notation

                return values

        return None

    def apply_jaw_transformations(self):
        """
        Apply transformation on loaded jaw model in the scene.
        All necessary information for the transformations are within the JawObject instance parameters.

        :param jaw: Instance of JawObject.

        :return: VTK 4x4 transformation instance needed for the jaw translation.
        """

        transform = vtk.vtkTransform()

        '''Translation to the object mass center'''
        transform.Translate(self.mass_center[0], self.mass_center[1], self.mass_center[2])

        if False:
            '''Applying jaw transformation matrix trom given XML file'''
            transform.Concatenate(self.transformation_matrix)

            '''Rotation of upper jaws to be in the same position as lower'''
            if self.jaw_type == 'upper':
                transform.RotateWXYZ(180, 0, 1, 0)

        '''Translate by the mass center.'''
        transform.Translate(-self.mass_center[0], -self.mass_center[1], -self.mass_center[2])

        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputData(self.poly)
        transformFilter.Update()

        mpr = vtk.vtkPolyDataMapper()
        mpr.SetInputConnection(transformFilter.GetOutputPort())

        self.actor.SetMapper(mpr)

        return transform

    @staticmethod
    def get_transformation_matrix_from_xml(xml_path, model_number):
        """
        Method extracts data from an XML file. These data form a transformation matrix that is later
        used for the polygon transformation.

        :param xml_path: Path to the XML file from which the data are to be extracted.
        :param model_number: Within the XML file, matrices for more model numbers are present.
                             This parameter specifies which one should be parsed.

        :return: VTK transformation matrix to be applied on the 3D model.
        """

        try:
            '''Find the XML tree root.'''
            root = ElementTree.parse(xml_path).getroot()

            '''Extract models from the XML.'''
            models_in_xml = root.findall('Model')

            '''Find the right Model within the tree.'''
            my_matrix = models_in_xml[int(model_number)].find('TransformationMatrix').text.split(' ')

            transformation_matrix = vtk.vtkMatrix4x4()

            '''Parse the values into matrix elements.'''
            for i in range(4):
                for j in range(4):
                    transformation_matrix.SetElement(i, j, float(my_matrix[i * 4 + j]))

            '''Expand the matrix with for easier matrix operations.'''
            transformation_matrix = expand_affine_transformation_r_c(transformation_matrix)
            transformation_matrix.Transpose()

        except (OSError, ElementTree.ParseError) as _:
            transformation_matrix = vtk.vtkMatrix4x4()

            for i in range(4):
                for j in range(4):
                    transformation_matrix.SetElement(i, j, float(0))

            transformation_matrix = expand_affine_transformation_r_c(transformation_matrix)
            transformation_matrix.Transpose()

        return transformation_matrix
