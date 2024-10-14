import re

def extract_number(name):
    # Find the first sequence of digits in the string
    match = re.search(r'\d+', name)
    if match:
        return int(match.group())
    return None

name = "train-555-of-40000blablaotherthings"
number = extract_number(name)
print(number)  # Output: 5
