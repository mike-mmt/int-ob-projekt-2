function(e) {    
    // Q-Table and helper functions translated from Python to JavaScript
    var q_table = ;

    function calculate_angle(x1, y1, x2, y2) {
        return (360 + Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI) % 360;
    }

    function calculate_angle_difference(desired_angle, tank_rotation) {
        return ((desired_angle - tank_rotation) + 180) % 360 - 180;
    }

    function get_state(x, y, angle_diff) {
        const GRID_SIZE = 10;  // Same as used in Python code
        const ANGLE_BIN_SIZE = 10;
        const MIN_X = 15.0;
        const MAX_X = 485.0;
        const MIN_Y = 15.0;
        const MAX_Y = 485.0;
        
        let x_discrete = Math.min(Math.floor((x - MIN_X) / GRID_SIZE), (MAX_X - MIN_X) / GRID_SIZE - 1);
        let y_discrete = Math.min(Math.floor((y - MIN_Y) / GRID_SIZE), (MAX_Y - MIN_Y) / GRID_SIZE - 1);
        let angle_discrete = Math.floor(angle_diff / ANGLE_BIN_SIZE) + 18;
        return [x_discrete, y_discrete, angle_discrete];
    }

    function choose_action(state) {
        let [x, y, angle] = state;
        let action_index = q_table[x][y][angle].indexOf(Math.max(...q_table[x][y][angle]));
        return action_index;
    }

    // Tank's state
    let myTank = e.data.myTank;
    let x = myTank.x;
    let y = myTank.y;
    let rotation = myTank.rotation;

    // Desired state (center of the stage)
    const CENTER_X = 250.0;
    const CENTER_Y = 250.0;

    let desired_angle = calculate_angle(x, y, CENTER_X, CENTER_Y);
    let angle_diff = calculate_angle_difference(desired_angle, rotation);
    let state = get_state(x, y, angle_diff);

    // Choose action based on Q-Table
    let action = choose_action(state);

    // Construct response
    var response = {};
    response.turnLeft = action === 0 ? 1 : 0;
    response.turnRight = action === 1 ? 1 : 0;
    response.goForward = action === 2 ? 1 : 0;

    // Shooting control (optional, based on your needs)
    if (e.data.myTank.shootCooldown == 0) {
        response.shoot = 1;
    }

    self.postMessage(response);
}
