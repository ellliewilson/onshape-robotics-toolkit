"""
This module contains functions to generate URDF components from Onshape assembly data.

"""

import os
import random
from typing import Optional, Union

import numpy as np
from networkx import DiGraph

from onshape_robotics_toolkit.connect import Asset, Client
from onshape_robotics_toolkit.log import LOGGER
from onshape_robotics_toolkit.models.assembly import (
    MateFeatureData,
    MateRelationFeatureData,
    MateType,
    Part,
)
from onshape_robotics_toolkit.models.document import WorkspaceType
from onshape_robotics_toolkit.models.geometry import MeshGeometry
from onshape_robotics_toolkit.models.joint import (
    BaseJoint,
    DummyJoint,
    FixedJoint,
    JointLimits,
    # JointDynamics,
    JointMimic,
    PrismaticJoint,
    RevoluteJoint,
)
from onshape_robotics_toolkit.models.link import (
    Axis,
    CollisionLink,
    Colors,
    Inertia,
    InertialLink,
    Link,
    Material,
    Origin,
    VisualLink,
)
from onshape_robotics_toolkit.parse import CHILD, MATE_JOINER, PARENT
from onshape_robotics_toolkit.utilities.helpers import get_sanitized_name

SCRIPT_DIR = os.path.dirname(__file__)


def get_joint_name(mate_id: str, mates: dict[str, MateFeatureData]) -> str:
    """
    Get the name of the joint from the mate id.

    Args:
        mate_id: The id of the mate.
        mates: The dictionary of mates in the assembly.

    Returns:
        The name of the joint.
    """
    reverse_mates = {mate.id: key for key, mate in mates.items()}
    return reverse_mates.get(mate_id)


def get_robot_link(
    name: str,
    part: Part,
    wid: str,
    client: Client,
    mate: Optional[Union[MateFeatureData, None]] = None,
    up_axis: str = 'z',
) -> tuple[Link, np.matrix, Asset]:
    """
    Generate a URDF link from an Onshape part with specified up axis.

    Args:
        name: The name of the link.
        part: The Onshape part object.
        wid: The unique identifier of the workspace.
        client: The Onshape client object to use for sending API requests.
        mate: MateFeatureData object to use for generating the transformation matrix.
        up_axis: 'z' for Z-up or 'y' for Y-up coordinate system.

    Returns:
        tuple[Link, np.matrix, Asset]: The generated link object,
            the transformation matrix from the STL origin to the link origin,
            and the asset.
    """
    # rotation_matrix = np.eye(4)
    _link_to_stl_tf = np.eye(4)

    if mate is None:
        # for root link, get the COM and use that as the origin for stl
        # rotation_matrix = get_up_axis_rotation(up_axis)
        _link_to_stl_tf[:3, 3] = np.array(part.MassProperty.center_of_mass).reshape(3)
    elif mate.matedEntities[CHILD].parentCS:
        # for non-root links, get the parentCS and transform it to the mateCS
        _link_to_stl_tf = mate.matedEntities[CHILD].parentCS.part_tf @ mate.matedEntities[CHILD].matedCS.part_to_mate_tf
    else:
        _link_to_stl_tf = mate.matedEntities[CHILD].matedCS.part_to_mate_tf

    # Apply rotation based on up_axis
    _link_to_stl_tf = _link_to_stl_tf

    _stl_to_link_tf = np.matrix(np.linalg.inv(_link_to_stl_tf))
    _mass = part.MassProperty.mass[0]
    _origin = Origin.zero_origin()
    _com = part.MassProperty.center_of_mass_wrt(_stl_to_link_tf)
    _inertia = part.MassProperty.inertia_wrt(np.matrix(_stl_to_link_tf[:3, :3]))
    _principal_axes_rotation = (0.0, 0.0, 0.0)

    LOGGER.info(f"Creating robot link for {name} with up_axis={up_axis}")

    if part.documentVersion:
        wtype = WorkspaceType.V.value
        mvwid = part.documentVersion

    elif part.isRigidAssembly:
        wtype = WorkspaceType.W.value
        mvwid = part.rigidAssemblyWorkspaceId
    else:
        wtype = WorkspaceType.W.value
        mvwid = wid

    _asset = Asset(
        did=part.documentId,
        wtype=wtype,
        wid=mvwid,
        eid=part.elementId,
        partID=part.partId,
        client=client,
        transform=_stl_to_link_tf,
        is_rigid_assembly=part.isRigidAssembly,
        file_name=f"{name}.stl",
    )

    _mesh_path = _asset.relative_path

    _link = Link(
        name=name,
        visual=VisualLink(
            name=f"{name}_visual",
            origin=_origin,
            geometry=MeshGeometry(_mesh_path),
            material=Material.from_color(name=f"{name}-material", color=random.SystemRandom().choice(list(Colors))),
        ),
        inertial=InertialLink(
            origin=Origin(
                xyz=_com,
                rpy=_principal_axes_rotation,
            ),
            mass=_mass,
            inertia=Inertia(
                ixx=_inertia[0, 0],
                ixy=_inertia[0, 1],
                ixz=_inertia[0, 2],
                iyy=_inertia[1, 1],
                iyz=_inertia[1, 2],
                izz=_inertia[2, 2],
            ),
        ),
        collision=CollisionLink(
            name=f"{name}_collision",
            origin=_origin,
            geometry=MeshGeometry(_mesh_path),
        ),
    )

    return _link, _stl_to_link_tf, _asset


def get_robot_joint(
    parent: str,
    child: str,
    mate: MateFeatureData,
    stl_to_parent_tf: np.matrix,
    mimic: Optional[JointMimic] = None,
    is_rigid_assembly: bool = False,
    up_axis: str = 'z',
) -> tuple[list[BaseJoint], Optional[list[Link]]]:
    """
    Generate a URDF joint from an Onshape mate feature with specified up axis.

    Args:
        parent: The name of the parent link.
        child: The name of the child link.
        mate: The Onshape mate feature object.
        stl_to_parent_tf: The transformation matrix from the STL origin to the parent link origin.
        mimic: The mimic joint object.
        is_rigid_assembly: Whether the assembly is a rigid assembly.
        up_axis: 'z' for Z-up or 'y' for Y-up coordinate system.

    Returns:
        tuple[list[BaseJoint], Optional[list[Link]]]: The generated joint object and the links.
    """
    # rotation_matrix = get_up_axis_rotation(up_axis)
    # rotation_matrix = get_rotation_matrix(np.pi / 2, "x")
    # rotation_matrix = np.eye(4)

    links = []
    if isinstance(mate, MateFeatureData):
        if not is_rigid_assembly:
            parent_to_mate_tf = mate.matedEntities[PARENT].matedCS.part_to_mate_tf
        else:
            # for rigid assemblies, get the parentCS and transform it to the mateCS
            parent_to_mate_tf = (
                mate.matedEntities[PARENT].parentCS.part_tf @ mate.matedEntities[PARENT].matedCS.part_to_mate_tf
            )

    stl_to_mate_tf = stl_to_parent_tf @ parent_to_mate_tf
    # Apply rotation based on up_axis
    origin = Origin.from_matrix(stl_to_mate_tf)
    # origin.rpy = (0.0, 0.0, 0.0)

    sanitized_name = get_sanitized_name(mate.name)

    LOGGER.info(f"Creating robot joint from {parent} to {child} with up_axis={up_axis}")

    # Define axis transformations based on up_axis
    def transform_axis(axis_vector: tuple[float, float, float], up_axis: str, joint_type: MateType) -> tuple[float, float, float]:
        """
        Transforms the axis vector based on the up_axis and joint type.

        Args:
            axis_vector: The original axis vector.
            up_axis: 'z' for Z-up or 'y' for Y-up coordinate system.
            joint_type: The type of joint (REVOLUTE, SLIDER, etc.)

        Returns:
            Transformed axis vector.
        """
        if up_axis.lower() == 'y':
            if joint_type in [MateType.REVOLUTE, MateType.SLIDER, MateType.CYLINDRICAL]:
                # For Y-up, all rotational and translational joints should have Y as their axis
                return (0.0, 1.0, 0.0)
            elif joint_type == MateType.BALL:
                # For ball joints, maintain orthogonal axes but rotated to Y-up
                x, y, z = axis_vector
                y_new = z
                z_new = -y
                return (x, y_new, z_new)

        return axis_vector

    if mate.mateType == MateType.REVOLUTE:
        transformed_axis = transform_axis((0.0, 0.0, -1.0), up_axis, mate.mateType)
        return [
            RevoluteJoint(
                name=sanitized_name,
                parent=parent,
                child=child,
                origin=origin,
                limits=JointLimits(
                    effort=1.0,
                    velocity=1.0,
                    lower=-2 * np.pi,
                    upper=2 * np.pi,
                ),
                axis=Axis(transformed_axis),
                # dynamics=JointDynamics(damping=0.1, friction=0.1),
                mimic=mimic,
            )
        ], links

    elif mate.mateType == MateType.FASTENED:
        return [FixedJoint(name=sanitized_name, parent=parent, child=child, origin=origin)], links

    elif mate.mateType == MateType.SLIDER or mate.mateType == MateType.CYLINDRICAL:
        transformed_axis = transform_axis((0.0, 0.0, -1.0), up_axis, mate.mateType)
        return [
            PrismaticJoint(
                name=sanitized_name,
                parent=parent,
                child=child,
                origin=origin,
                # limits=JointLimits(
                #     effort=1.0,
                #     velocity=1.0,
                #     lower=-0.1,
                #     upper=0.1,
                # ),
                axis=Axis(transformed_axis),
                # dynamics=JointDynamics(damping=0.1, friction=0.1),
                mimic=mimic,
            )
        ], links

    elif mate.mateType == MateType.BALL:
        dummy_x = Link(
            name=f"{parent}_{get_sanitized_name(mate.name)}_x",
            inertial=InertialLink(
                mass=0.0,
                inertia=Inertia.zero_inertia(),
                origin=Origin.zero_origin(),
            ),
        )
        dummy_y = Link(
            name=f"{parent}_{get_sanitized_name(mate.name)}_y",
            inertial=InertialLink(
                mass=0.0,
                inertia=Inertia.zero_inertia(),
                origin=Origin.zero_origin(),
            ),
        )

        links = [dummy_x, dummy_y]

        # Define transformed axes for ball joints
        transformed_axis_x = transform_axis((1.0, 0.0, 0.0), up_axis, mate.mateType)
        transformed_axis_y = transform_axis((0.0, 1.0, 0.0), up_axis, mate.mateType)
        transformed_axis_z = transform_axis((0.0, 0.0, -1.0), up_axis, mate.mateType)

        return [
            RevoluteJoint(
                name=sanitized_name + "_x",
                parent=parent,
                child=dummy_x.name,
                origin=origin,
                limits=JointLimits(
                    effort=1.0,
                    velocity=1.0,
                    lower=-2 * np.pi,
                    upper=2 * np.pi,
                ),
                axis=Axis(transformed_axis_x),
                # dynamics=JointDynamics(damping=0.1, friction=0.1),
                mimic=mimic,
            ),
            RevoluteJoint(
                name=sanitized_name + "_y",
                parent=dummy_x.name,
                child=dummy_y.name,
                origin=Origin.zero_origin(),
                limits=JointLimits(
                    effort=1.0,
                    velocity=1.0,
                    lower=-2 * np.pi,
                    upper=2 * np.pi,
                ),
                axis=Axis(transformed_axis_y),
                # dynamics=JointDynamics(damping=0.1, friction=0.1),
                mimic=mimic,
            ),
            RevoluteJoint(
                name=sanitized_name + "_z",
                parent=dummy_y.name,
                child=child,
                origin=Origin.zero_origin(),
                limits=JointLimits(
                    effort=1.0,
                    velocity=1.0,
                    lower=-2 * np.pi,
                    upper=2 * np.pi,
                ),
                axis=Axis(transformed_axis_z),
                # dynamics=JointDynamics(damping=0.1, friction=0.1),
                mimic=mimic,
            ),
        ], links

    else:
        LOGGER.warning(f"Unsupported joint type: {mate.mateType}")
        return [DummyJoint(name=sanitized_name, parent=parent, child=child, origin=origin)], links


def get_topological_mates(
    graph: DiGraph,
    mates: dict[str, MateFeatureData],
    relations: Optional[dict[str, MateRelationFeatureData]] = None,
) -> tuple[dict[str, MateFeatureData], dict[str, MateRelationFeatureData]]:
    """
    Get the topological mates from the graph. This shuffles the order of the mates to match the directed graph edges.

    Args:
        graph: The graph representation of the assembly.
        mates: The dictionary of mates in the assembly.
        relations: The dictionary of relations in the assembly.

    Returns:
        tuple[dict[str, MateFeatureData], dict[str, MateRelationFeatureData]]: The topological mates and relations.

    Examples:
        >>> get_topological_mates(graph, mates)
        (
            {
                'link1-MATE-body': MateFeatureData(...),
                'subassembly1-SUB-link2-MATE-body': MateFeatureData(...),
            },
            {
                'link1-MATE-body-REL-subassembly1-SUB-link2-MATE-body': MateRelationFeatureData(...),
            }
        )
    """
    topological_mates: dict[str, MateFeatureData] = {}
    topological_relations: dict[str, MateRelationFeatureData] = relations or {}

    mate_keys = {tuple(key.split(MATE_JOINER)) for key in mates}
    graph_edges = set(graph.edges)

    rogue_mates = mate_keys.difference(graph_edges)

    for edge in graph.edges:
        parent, child = edge
        key = f"{parent}{MATE_JOINER}{child}"

        if (child, parent) in rogue_mates:
            # the only way it can be a rogue mate is if the parent and child are swapped
            # LOGGER.info(f"Rogue mate found: {edge}")
            rogue_key = f"{child}{MATE_JOINER}{parent}"
            topological_mates[key] = mates[rogue_key]

            if isinstance(topological_mates[key], MateFeatureData):
                topological_mates[key].matedEntities = topological_mates[key].matedEntities[::-1]

            if relations and rogue_key in topological_relations:
                LOGGER.info(f"Rogue relation found: {rogue_key}")
                topological_relations[key] = topological_relations[rogue_key]
                topological_relations.pop(rogue_key)

        else:
            topological_mates[key] = mates[key]

    return topological_mates, topological_relations

def get_rotation_matrix(angle: float, axis: str) -> np.ndarray:
    if axis == "x":
        return np.array([
                [1, 0, 0, 0],
                [0, np.cos(angle), -np.sin(angle), 0],
                [0, np.sin(angle), np.cos(angle), 0],
                [0, 0, 0, 1],
            ]
        )

    elif axis == "y":
        return np.array(
            [
                [np.cos(angle), 0, np.sin(angle), 0],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1],
            ]
        )

    elif axis == "z":
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )


def get_up_axis_rotation(up_axis: str) -> np.matrix:
    """
    Returns the rotation matrix based on the up_axis parameter.

    Args:
        up_axis: 'z' for Z-up, 'y' for Y-up.

    Returns:
        A 4x4 rotation matrix.
    """
    rotation = get_rotation_matrix(-np.pi / 2, "x") if up_axis.lower() == "y" else np.eye(4)
    return np.matrix(rotation)
