
def format_seconds(duration):
    hours, remainder = divmod(duration, 3600)
    days, hours = divmod(hours, 24)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        return f"{int(days)} days, {int(hours)}:{int(minutes)}:{int(seconds)} hours"
    
    if hours > 0:
        return f"{int(hours)}:{int(minutes)}:{int(seconds)} hours"
    
    if minutes > 0:
        return f"{int(minutes)}:{int(seconds)} minutes"
    
    return f"{int(seconds)} seconds"