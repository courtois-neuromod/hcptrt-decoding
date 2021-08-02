def sum_(arr): 
    sum = 0
    for i in arr:
        sum = sum + i
    return(sum) 

'--------------------------------------------------------------------------------'

def between(value, before, after):
    """
    Parameters
    ----------
    value: str
        File name which is a BIDS-formatted in the CNeuroMod datasets,
        for which to extract a value between two characters.
    before: str
        First charcter or part of the file name
    after: str
        First charcter or part of the file name
    """
    
    # Find and validate before-part
    pos_before = value.find(before)
    if pos_before == -1: return 
    
    # Find and validate after part
    pos_after = value.find(after)
    if pos_after == -1: return
    
    # Return middle part
    adjusted_pos_before = pos_before + len(before)
    if adjusted_pos_before >= pos_after: return 
    return value[adjusted_pos_before:pos_after]