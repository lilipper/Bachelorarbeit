from pathlib import Path

def print_tree(root=".", max_depth=None, show_files=True, ignore_hidden=True, prefix=""):
    root = Path(root)
    if ignore_hidden and root.name.startswith("."):
        return

    print(prefix + root.name + ("/" if root.is_dir() else ""))
    if not root.is_dir():
        return

    if max_depth is not None and max_depth <= 0:
        return

    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    if not show_files:
        entries = [e for e in entries if e.is_dir()]
    if ignore_hidden:
        entries = [e for e in entries if not e.name.startswith(".")]

    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        next_prefix = prefix + ("    " if i == len(entries) - 1 else "│   ")
        print(prefix + connector + entry.name + ("/" if entry.is_dir() else ""))
        if entry.is_dir():
            print_tree(
                entry,
                None if max_depth is None else max_depth - 1,
                show_files=show_files,
                ignore_hidden=ignore_hidden,
                prefix=next_prefix
            )

# Beispiele:
print_tree(".")  