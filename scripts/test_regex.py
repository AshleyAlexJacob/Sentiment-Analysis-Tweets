import re

text = "Hello!!! Naveed Af$ridi This is @@sample### text $$$ with ^^^ extra *** symbols???"

# Regex to keep only letters, numbers, and spaces
cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)

print("Original:", text)
print("Cleaned :", cleaned)
