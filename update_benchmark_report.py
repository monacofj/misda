
import json

path = "benchmark.ipynb"

with open(path, "r") as f:
    nb = json.load(f)

# Goal:
# Replace the blocks "Print Detailed Structure" and "Print Ranked MIS Details"
# with a single call to print(result.report())

modified = False

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    
    source = cell["source"]
    new_source = []
    
    skip = False
    replaced = False

    for line in source:
        # Detect start of verbose sections
        if "# --- 2. Print Detailed Structure" in line:
            skip = True
            if not replaced:
                 new_source.append("        # --- 2. Full Technical Report ---\n")
                 new_source.append("        print(result.report())\n")
                 replaced = True
        
        # Detect end of verbose sections (start of Plot Graph)
        if "# --- 4. Plot Graph" in line:
            skip = False
        
        if not skip:
            new_source.append(line)
    
    if len(new_source) != len(source):
        cell["source"] = new_source
        modified = True

if modified:
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Updated benchmark.ipynb to use result.report()")
else:
    print("No changes needed (sections not found as expected)")
