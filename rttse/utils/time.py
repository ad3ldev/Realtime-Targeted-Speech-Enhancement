
def format_seconds(duration):
    hours, remainder = divmod(duration, 3600)
    days, hours = divmod(hours, 24)
    minutes, seconds = divmod(remainder, 60)
    if days == 0:
        if hours == 0:
            eta = f"{int(minutes)}:{int(seconds)} minutes"
            if minutes == 0:
                eta = f"{int(seconds)} seconds"
        else:
            eta = f"{int(hours)}:{int(minutes)}:{int(seconds)} hours"

    else:
        eta = f"{days} day, {int(hours)}:{int(minutes)}:{int(seconds)} hours"

    return eta