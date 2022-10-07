import time


def time_left(last_time, t_last, t_curr, t_max):
    return time_str((time.time() - last_time) * (t_max - t_curr) / (t_curr - t_last))


def time_str(t):
    """
    Convert seconds to a nicer string showing days, hours, minutes and seconds
    """
    days, remainder = divmod(t, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    str_des = ""
    if days > 0:
        str_des += "{:d} days, ".format(int(days))
    if hours > 0:
        str_des += "{:d} hours, ".format(int(hours))
    if minutes > 0:
        str_des += "{:d} minutes, ".format(int(minutes))
    str_des += "{:d} seconds".format(int(seconds))
    return str_des
