ALWAYS use the embodied tool for any robot, arm, serial, USB, motor, camera, or hardware question.
NEVER use exec to inspect /dev, serial devices, or raw hardware paths.
ALWAYS start hardware questions by calling embodied(action="setup_show").
ALWAYS use embodied(action="identify") when the user wants to connect or name arms.
ALWAYS follow this workflow order: identify -> calibrate -> teleoperate -> record -> train -> run_policy.
ALWAYS refer to arms by aliases from setup. NEVER expose raw /dev paths to the user.
ALWAYS suggest the next step in text. NEVER auto-execute calibrate, teleoperate, or record without explicit user request.
NEVER call calibrate, teleoperate, or record unless user explicitly asks.
ALWAYS use follower_names and leader_names with arm aliases for teleoperate and record.
ALWAYS use structured setup actions (set_arm, rename_arm, remove_arm, set_camera, remove_camera) to change config.
NEVER auto-correct or normalize arm aliases.
NEVER ask the user to type raw serial device paths when setup already has scanned ports.
For record: provide dataset_name, task description, and num_episodes.
For train: provide dataset_name and optionally steps and device.
For run_policy: provide checkpoint_path or let it auto-detect the latest.
