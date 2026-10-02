import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The pattern looks for spaces followed by `index=safe_index(..., st.session_state["..."]),` and a newline.
# We will just replace it with an empty string.
new_content = re.sub(r'^\s*index=safe_index\(.*?\),\s*\n', '', content, flags=re.MULTILINE)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
    
print("Replaced instances.")
