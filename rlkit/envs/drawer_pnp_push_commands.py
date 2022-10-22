drawer_pnp_push_commands = [
    ### Task A ###
    {
        "init_pos": [0.8007752866589611, 0.1648154585720787, -0.14012390258012603],
        "drawer_open": False,
        "drawer_yaw": 11.04125081594207,
        "drawer_quadrant": 0,
        "small_object_pos": [0.70574489, 0.22969248, -0.35201094],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 1,
        "command_sequence": [
            ("move_obj_slide",
                {
                    "target_quadrant": 2,
                }
             ),
            ("move_drawer", {}),
        ],
        "drawer_hack": True,
        "drawer_hack_quadrant": 1,
    },
    ### Task B ###
    {
        "drawer_open": True,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_slide",
                {
                    "target_quadrant": 3,
                }
             )
        ],
    },
    ### Task C ###
    {
        "init_pos": [0.5559647343004194, -0.027816898388712933, -0.28124351873499637],
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.54798087, -0.006632, -0.34451221],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             ),
            ("move_drawer", {}),
        ],
    },
]
