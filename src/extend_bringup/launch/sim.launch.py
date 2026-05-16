import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scene_arg = DeclareLaunchArgument(
        'scene',
        default_value='gripper',
        description='Scene to load: gripper or xhand'
    )

    scene = LaunchConfiguration('scene')
    use_viewer = LaunchConfiguration('use_viewer')

    sim_node = Node(
        package='extend_sim',
        executable='sim_node',
        name='extend_sim_node',
        parameters=[{'scene': scene}],
        output='screen',
    )

    viewer_node = Node(
        package='extend_sim',
        executable='viewer_node',
        name='extend_viewer_node',
        parameters=[{'scene': scene}],
        output='screen',
        condition=launch.conditions.IfCondition(use_viewer)
    )

    recorder_node = Node(
        package='extend_recorder',
        executable='recorder_node',
        name='extend_recorder_node',
        parameters=[{'scene': scene}],
        output='screen',
    )

    return LaunchDescription([
        scene_arg,
        DeclareLaunchArgument(
            'use_viewer',
            default_value='true',
            description='Whether to launch the MuJoCo viewer'
        ),
        sim_node,
        viewer_node,
        recorder_node,
    ])
