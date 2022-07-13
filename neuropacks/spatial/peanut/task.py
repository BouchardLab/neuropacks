def forkmaze_task_reward_rule(rule, start_well, end_well, latest_visit):
    '''
    input:
        rule (str): either 'centerAlternation' or 'handleAlternation'
        start_well, end_well (char): single character code for a well
        latest_visit (char or None): either a well code, or None
    return:
        outcome_key (int): a code for outcome type for the trial (trajectory)
        outcome_keymap (dict): provides description for each outcome key
    '''
    # fixed_well: 'C' for centerAlternation, or 'H' for handleAlternation
    fixed_well = rule[0].upper()
    alt_wells=('L', 'R')

    # animal gets reward if outcome_key > 0
    outcome_keymap = {
        2: 'correct fixed_well from an alt_well',
        1: 'correct alt_well from fixed_well',
        0: 'incorrect alt_well from fixed_well',
        -1: 'incorrect pair - between the alt_wells',
        -2: 'incorrect pair - only end well in-task',
        -3: 'incorrect pair - only start well in-task',
        -4: 'incorrect pair - neither well in-task',    # this cannot happen
        -9: 'incorrect - returned to the same well'
    }

    if end_well == start_well:
        outcome_key = -9
    elif end_well == fixed_well:
        if start_well in alt_wells:
            outcome_key = 2
        else:
            outcome_key = -2
    elif end_well in alt_wells:
        if start_well == fixed_well:
            if end_well == latest_visit:
                outcome_key = 0
            else:
                outcome_key = 1
        elif start_well in alt_wells:
            outcome_key = -1
        else:
            outcome_key = -2
    elif start_well == fixed_well or start_well in alt_wells:
        # at this point, end_well is an out-of-task well
        outcome_key = -3
    else:
        # both end_well and start_well are out-of-task wells
        # (although in the current forkmaze setting, there will not happen
        # because there can be only one out-of-task well)
        outcome_key = -4

    # outcome_desc = outcome_keymap[outcome_key]
    return outcome_key, outcome_keymap


def get_traj_outcomes(traj_types, rule):
    '''
    input:
        traj_types (list): each item is 2-character string, like 'CR' or 'LH'
        rule (str): either 'centerAlternation' or 'handleAlternation'
    return:
        traj_outcomes (list): list of integer keys for task outcome types
    '''
    latest_visit = None
    traj_outcomes = []
    for traj_type in traj_types:
        # traj_type is a 2-character string
        start_well = traj_type[0]
        end_well = traj_type[1]
        outcome_key, _ = forkmaze_task_reward_rule(
            rule, start_well, end_well, latest_visit)
        traj_outcomes.append(outcome_key)
        latest_visit = end_well
    return traj_outcomes
